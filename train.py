import argparse
import json
import math
import multiprocessing
import os
import hashlib
from datetime import datetime

import numpy
import pandas as pd
import spacy
from BitVector import BitVector
from dask import delayed, compute
from textblob import TextBlob

from utils import dataset_utils, sentiment_utils

SENTIMENT_PHRASES_FILE_NAME = 'train_sentiment_phrases.json'
DONT_USE_OPTION = 0  # Don't produce the data associated with the optional command line argument.
USE_EXISTING = 1  # Produce the data associated with the optional command line argument and
# but use existing data if it exists.
CREATE_FROM_SCRATCH = 2  # Produce the data associated with the optional command line argument


# from scratch, even if it already exists in the respective output file.

class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def decrement(self):
        self.count -= 1

    def get_count(self):
        return self.count

    def set_count(self, new_count):
        self.count = new_count


def swap_columns(columns, source_label, counter):
    dest_index = counter.get_count()

    if source_label in columns and dest_index < len(columns):
        dest_index_label = columns[dest_index]
        source_label_index = columns.index(source_label)
        columns[dest_index] = source_label
        columns[source_label_index] = dest_index_label

    counter.increment()


def arrange_columns(data_set_dataframe, use_movie_ratings_distribution,
                    use_review_ratings_distribution, use_word_embeddings,
                    use_sentiment_phrases, use_sentence_sentiment, use_sentence_word_embeddings):
    columns = list(data_set_dataframe.columns)
    counter = Counter()

    if use_movie_ratings_distribution != DONT_USE_OPTION:
        swap_columns(columns, 'one_rating_num', counter)
        swap_columns(columns, 'two_rating_num', counter)
        swap_columns(columns, 'three_rating_num', counter)
        swap_columns(columns, 'four_rating_num', counter)
        swap_columns(columns, 'five_rating_num', counter)
    if use_review_ratings_distribution != DONT_USE_OPTION:
        swap_columns(columns, 'average_one_rating_phrase_percent', counter)
        swap_columns(columns, 'average_two_rating_phrase_percent', counter)
        swap_columns(columns, 'average_three_rating_phrase_percent', counter)
        swap_columns(columns, 'average_four_rating_phrase_percent', counter)
        swap_columns(columns, 'average_five_rating_phrase_percent', counter)
    if use_sentiment_phrases != DONT_USE_OPTION:
        swap_columns(columns, 'sentiment_phrases_found', counter)
    if use_word_embeddings != DONT_USE_OPTION:
        for i in range(0, 300):
            swap_columns(columns, 'vector_embedding_comp_' + str(i), counter)
    if use_sentence_sentiment != DONT_USE_OPTION or use_sentence_word_embeddings != DONT_USE_OPTION:
        swap_columns(columns, 'sentences_data', counter)
    swap_columns(columns, 'rating_label', counter)
    return data_set_dataframe[columns]


def write_sentiment_phrases_to_disk(sentiment_phrases):
    """
    Writes the sentiment phrases computed based on the training data set in a formatted fashion.
    :param sentiment_phrases: the sentiment phrases computed based on the training set.
    """
    with open(SENTIMENT_PHRASES_FILE_NAME, 'w') as output_sentiment_phrases_file:
        json.dump(sentiment_phrases, output_sentiment_phrases_file)


def compute_sentiment_phrases(input_train_data_set):
    """
    Identifies the sentiment phrases in the concatenation of all reviews of eacj movie,
    updates the count of sentiment phrases accordingly, and returns it.
    :param input_train_data_set: the list of movies and associated data in the training data set.
    :return: the dictionary of counts of sentiment phrases based on the training data set.
    """
    sentiment_phrases = {}
    counter = 0

    for movie in input_train_data_set:
        review_concatenation = "\n".join([review['text'] for review in movie['reviews']])
        curr_sentiment_phrases = [" ".join(assessment[0]) for assessment
                                  in TextBlob(review_concatenation).sentiment_assessments.assessments]
        for phrase in curr_sentiment_phrases:
            if phrase not in sentiment_phrases.keys():
                sentiment_phrases[phrase] = 1
            else:
                sentiment_phrases[phrase] += 1
        counter += 1
        print("{}: Number of movies processed for sentiment phrases: {}".format(datetime.now(), counter))
    return sentiment_phrases


def select_top_records(input_dictionary, top_percentage_to_use, filter_low_subjectivity_phrases=True):
    """
    Receives a dictionary of records of the form {key: integer}, sorts the dictionary by value
    in descending order, removes records whose value is below the (1-top_percentage_to_use)
    percentile, and returns the resulting dictionary.

    :param input_dictionary: a dictionary as described above.
    :param top_percentage_to_use: the top percentage to use to filter records.
                                  Only records whose value is below no more than the input
                                  percentage of all records will be kept.
    :param filter_low_subjectivity_phrases: whether to keep only phrases with relatively high subjectivity.
    :return: the dictionary transformed as described above.
    """
    number_of_phrases_to_include = math.floor(len(input_dictionary.keys())
                                              *
                                              min(top_percentage_to_use, 1))
    if filter_low_subjectivity_phrases:
        input_dictionary = {key: input_dictionary[key] for key in input_dictionary.keys()
                            if TextBlob(key).sentiment.subjectivity > 0}
    ordered_sentiment_phrases = list(reversed(sorted((v, k) for (k, v) in input_dictionary.items())))
    return {phrase: count for (count, phrase) in ordered_sentiment_phrases[:number_of_phrases_to_include]}


def get_sentiment_phrases_found(sentiment_phrases_index, review_concatenation):
    """
    For the input concatenation of the reviews of a given movie, and a dictionary of
    phrases and their corresponding decided indices, checks for each sentiment phrase
    if it is present in the reviews of the movie.
    For each sentiment phrase that is found, a bit in the corresponding index is
    set to 1 in a generated bit vector.
    Finally, the method converts the bit vector into a string, hashes it and returns
    the hash value computed.
    :param sentiment_phrases_index: A dictionary between each sentiment phrase and
                                    an index assigned to it.
    :param review_concatenation: A concatenation of all reviews of a given movie.
    :return: a hash value representing which sentiment phrases are found in the reviews
             of the given movie.
    """
    bv = BitVector(bitlist=[0] * len(sentiment_phrases_index.keys()))
    m = hashlib.sha256()
    for phrase in sentiment_phrases_index.keys():
        if phrase in review_concatenation:
            bv[sentiment_phrases_index[phrase]] = 1
    m.update(str.encode(str(bv)))
    return int(m.hexdigest(), 16)


def create_sentiment_phrases_index_map(sentiment_phrases):
    key_list = list(sentiment_phrases.keys())
    return {key_list[i]: i for i in range(0, len(key_list))}


def get_sentence_data(sentence, movie_summary):
    """
    Receives a sentence and returns a tuple containing in the first component the sentence itself,
    and in the other two components the sentiment score of the sentence and the sum of embeddings of
    the words in the sentence.
    :param sentence: a sentence to process as described above.
    :return: a tuple containing data about the input sentence, as described above.
    """
    return [sentence,
            round(dataset_utils.shift_scale(TextBlob(sentence).sentiment.polarity, -1, 1, 1, 5)),
            [round(float(component), 3) for component in nlp(sentence).vector],
            dataset_utils.calculate_rouge(dataset_utils.preprocess_text(movie_summary),
                                          dataset_utils.preprocess_text(sentence), 1)['fScore']]


def process_movie(movie, dataset_name,
                  use_movie_rating_distribution,
                  use_review_rating_distribution, use_word_embeddings,
                  use_sentiment_phrases, use_sentence_sentiment, use_sentence_word_embeddings,
                  sentiment_phrases_index_map):
    sentences_data = []
    review_concatenation = ".".join([review['text'] for review in movie['reviews']])
    if use_sentence_sentiment == CREATE_FROM_SCRATCH or use_sentence_word_embeddings == CREATE_FROM_SCRATCH:
        curr_sentences = [sentence for sentence in review_concatenation.split(".")
                          if len(sentence) > 0]
        sentences_data.extend([result_component for result in
                               compute([delayed(get_sentence_data)(sentence, movie['summary'])
                               for sentence in curr_sentences])
                               for result_component in result[0]])
        movie['sentences_data'] = sentences_data
    if use_movie_rating_distribution == CREATE_FROM_SCRATCH \
            or \
            use_review_rating_distribution == CREATE_FROM_SCRATCH:
        sentiment_scores = [sentiment_utils.get_sentiment_score(review['text'], 1, 5)
                            for review in movie['reviews']]
    if use_movie_rating_distribution == CREATE_FROM_SCRATCH:
        movie['one_rating_num'] = len([score[0] for score in sentiment_scores if score[0] == 1])
        movie['two_rating_num'] = len([score[0] for score in sentiment_scores if score[0] == 2])
        movie['three_rating_num'] = len([score[0] for score in sentiment_scores if score[0] == 3])
        movie['four_rating_num'] = len([score[0] for score in sentiment_scores if score[0] == 4])
        movie['five_rating_num'] = len([score[0] for score in sentiment_scores if score[0] == 5])
    if use_review_rating_distribution == CREATE_FROM_SCRATCH:
        movie['average_one_rating_phrase_percent'] = sum([score[1] for score in sentiment_scores]) \
                                                     / \
                                                     len(movie['reviews'])
        movie['average_two_rating_phrase_percent'] = sum([score[2] for score in sentiment_scores]) \
                                                     / \
                                                     len(movie['reviews'])
        movie['average_three_rating_phrase_percent'] = sum([score[3] for score in sentiment_scores]) \
                                                       / \
                                                       len(movie['reviews'])
        movie['average_four_rating_phrase_percent'] = sum([score[4] for score in sentiment_scores]) \
                                                      / \
                                                      len(movie['reviews'])
        movie['average_five_rating_phrase_percent'] = sum([score[5] for score in sentiment_scores]) \
                                                      / \
                                                      len(movie['reviews'])
    if use_word_embeddings == CREATE_FROM_SCRATCH:
        reviews_average_vector_embedding = nlp(review_concatenation).vector
        # reviews_average_vector_embedding = nlp(review_concatenation).vector / len(movie['reviews'])
        for i in range(0, 300):
            movie['vector_embedding_comp_' + str(i)] = float(reviews_average_vector_embedding[i])
    if use_sentiment_phrases == CREATE_FROM_SCRATCH:
        movie['sentiment_phrases_found'] = get_sentiment_phrases_found(sentiment_phrases_index_map,
                                                                       review_concatenation)
    movie['rating_label'] = round(float(numpy.mean([review['rating']
                                                    for review in movie['reviews']])))
    print("{}: Dataset \'{}\': Processed movie".format(datetime.now(), dataset_name))
    # counter += 1
    # print("{}: Dataset \'{}\': Number of movies processed: {}".format(datetime.now(),
    #                                                                   dataset_name, counter))

    return movie


def construct_data_set(dataset_name, data_set_list_of_dicts, use_movie_rating_distribution,
                       use_review_rating_distribution, use_word_embeddings,
                       use_sentiment_phrases, use_sentence_sentiment, use_sentence_word_embeddings):
    counter = 1
    sentiment_phrases = dict()
    should_compute_sentiment_phrases = dataset_name == 'train' \
                                       and \
                                       not os.path.exists(SENTIMENT_PHRASES_FILE_NAME) \
                                       and \
                                       use_sentiment_phrases == CREATE_FROM_SCRATCH
    if not should_compute_sentiment_phrases and os.path.exists(SENTIMENT_PHRASES_FILE_NAME):
        print('{}: Start of loading of sentiment phrases.'.format(datetime.now()))
        with open(SENTIMENT_PHRASES_FILE_NAME, 'r') as input_file:
            sentiment_phrases = json.load(input_file)
            sentiment_phrases = select_top_records(sentiment_phrases, 0.15)
        print('{}: End of loading of sentiment phrases.'.format(datetime.now()))
    elif dataset_name == 'train':
        print('{}: Start of computation of sentiment phrases.'.format(datetime.now()))
        sentiment_phrases = compute_sentiment_phrases(data_set_list_of_dicts)
        write_sentiment_phrases_to_disk(sentiment_phrases)
        sentiment_phrases = select_top_records(sentiment_phrases, 0.15)
        print('{}: End of computation of sentiment phrases.'.format(datetime.now()))
    sentiment_phrases_index_map = create_sentiment_phrases_index_map(sentiment_phrases)

    results = []
    for movie in data_set_list_of_dicts:
        results.append(delayed(process_movie)(movie, dataset_name, use_movie_rating_distribution,
                                              use_review_rating_distribution, use_word_embeddings,
                                              use_sentiment_phrases, use_sentence_sentiment,
                                              use_sentence_word_embeddings,
                                              sentiment_phrases_index_map))
        print("{}: Added task for movie No. {}".format(datetime.now(), counter))
        counter += 1
    results = compute(results)[0]

    # for movie in data_set_list_of_dicts:

    return [movie for movie in results]


def fill_values(dataframe, column_name, column_values):
    dataframe[column_name] = column_values
    return dataframe


def dataset_exists(dataset_name):
    return os.path.exists("{}.json".format(dataset_name))


def get_data_set(dataset_name, use_movie_ratings_distribution,
                 use_review_ratings_distribution, use_word_embeddings,
                 use_sentiment_phrases, use_sentence_sentiment, use_sentence_word_embeddings):
    with open('{}.json'.format(dataset_name), 'r') as input_file:
        dataset = arrange_columns(pd.DataFrame(json.load(input_file)), use_movie_ratings_distribution,
                                  use_review_ratings_distribution, use_word_embeddings,
                                  use_sentiment_phrases, use_sentence_sentiment, use_sentence_word_embeddings)
    return dataset_utils.drop_redundant_columns(dataset)


parser = argparse.ArgumentParser(description='Creates two json files, a train.json file and a gold.json '
                                             +
                                             'file, containing the training and gold sets respectively.\n'
                                             'The program produces the files based on the input corpus '
                                             'and the input flags.\n'
                                             +
                                             'For each optional flag, 0 denotes not to use the option, '
                                             +
                                             '1 denotes to use the option and use the data in an '
                                             +
                                             'existing output dataset if it exists, and 2 denotes '
                                             +
                                             'to use the option and produce the data from scratch '
                                             +
                                             'even if it already exists in the output dataset.\n'
                                             +
                                             'If no options are explicitly set, they are all set by '
                                             +
                                             'default to 1.')
parser.add_argument("CORPUS_PATH",
                    help="The path of the complete corpus with associated metadata, as an"
                         +
                         " appropriately formatted json file.")
parser.add_argument("OUTPUT_FILES_PATH",
                    help="The directory in which to put the output files.")
parser.add_argument("-mrd", "--movie_rating_distribution", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Use the distribution of 1,2,3,4,5 ratings per movie.")
parser.add_argument("-rrd", "--review_rating_distribution", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Use the average percentages of 1,2,3,4,5 ratings per review.")
parser.add_argument("-we", "--word_embeddings", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Use word embeddings computed based on word2vec.")
parser.add_argument("-sp", "--sentiment_phrases", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Identify sentiment carrying phrases and for each movie's reviews, check "
                         "if they are found in them.")
parser.add_argument("-ts_ss", "--textual_summarization_sentence_sentiment", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Use the sentiment score of each sentence to rank it for deciding whether to "
                         +
                         "choose to include it in a textual summary.")
parser.add_argument("-ts_swe", "--textual_summarization_sentence_word_embeddings", type=int, choices=[0, 1, 2],
                    default=1,
                    help="Use the sum of word embeddings of each sentence to rank the sentence to decide whether "
                         +
                         "to choose to include it in a textual summary.")
args = parser.parse_args()
if not os.path.exists(args.CORPUS_PATH):
    print('Error: the file {} does not exist.'.format(args.CORPUS_PATH))
    exit(1)
elif not os.path.exists(args.OUTPUT_FILES_PATH):
    print('Error: the file {} does not exist.'.format(args.OUTPUT_FILES_PATH))
    exit(1)
elif not os.path.isdir(args.OUTPUT_FILES_PATH):
    print('Error: the argument OUTPUT_FILES_PATH must be a directory.'.format(args.OUTPUT_FILES_PATH))
    exit(1)

nlp = spacy.load('en_core_web_lg')
new_datasets_created = False

if args.movie_rating_distribution == USE_EXISTING \
        or \
        args.review_rating_distribution == USE_EXISTING \
        or \
        args.word_embeddings == USE_EXISTING \
        or \
        args.sentiment_phrases == USE_EXISTING \
        or \
        args.textual_summarization_sentence_sentiment == USE_EXISTING \
        or \
        args.textual_summarization_sentence_word_embeddings == USE_EXISTING:
    if not os.path.exists(os.path.join(args.OUTPUT_FILES_PATH, 'train.json')) \
            or \
            not os.path.exists(os.path.join(args.OUTPUT_FILES_PATH, 'gold.json')):
        print('Error: user chose to use existing data but at least one of the files '
              +
              'train.json or gold.json does not exist in {}.'.format(args.OUTPUT_FILES_PATH))
        exit(1)
    train_data_set = get_data_set(os.path.join(args.OUTPUT_FILES_PATH, 'train'),
                                  args.movie_rating_distribution,
                                  args.review_rating_distribution,
                                  args.word_embeddings, args.sentiment_phrases,
                                  args.textual_summarization_sentence_sentiment,
                                  args.textual_summarization_sentence_word_embeddings) \
        .to_dict('records')
    gold_data_set = get_data_set('gold', args.movie_rating_distribution,
                                 args.review_rating_distribution,
                                 args.word_embeddings, args.sentiment_phrases,
                                 args.textual_summarization_sentence_sentiment,
                                 args.textual_summarization_sentence_word_embeddings) \
        .to_dict('records')
else:
    data_sets = dataset_utils.build_data_sets_from_json_file(args.CORPUS_PATH)
    train_data_set = data_sets['train']
    gold_data_set = data_sets['gold']

train_data_set = construct_data_set('train', train_data_set,
                                    args.movie_rating_distribution,
                                    args.review_rating_distribution,
                                    args.word_embeddings, args.sentiment_phrases,
                                    args.textual_summarization_sentence_sentiment,
                                    args.textual_summarization_sentence_word_embeddings)
with open(os.path.join(args.OUTPUT_FILES_PATH, 'train.json'), 'w') as output_file:
    json.dump(train_data_set, output_file)
gold_data_set = construct_data_set('gold', gold_data_set,
                                   args.movie_rating_distribution,
                                   args.review_rating_distribution,
                                   args.word_embeddings, args.sentiment_phrases,
                                   args.textual_summarization_sentence_sentiment,
                                   args.textual_summarization_sentence_word_embeddings)
with open(os.path.join(args.OUTPUT_FILES_PATH, 'gold.json'), 'w') as output_file:
   json.dump(gold_data_set, output_file)
