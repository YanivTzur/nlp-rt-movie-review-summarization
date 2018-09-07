import json
import os
import sys
import math
from datetime import datetime

import numpy
import pandas as pd
import spacy
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
from BitVector import BitVector

from utils import dataset_utils, sentiment_utils

SENTIMENT_PHRASES_FILE_NAME = 'train_sentiment_phrases.json'


def swap_columns(columns, source_label, dest_index):
    dest_index_label = columns[dest_index]
    source_label_index = columns.index(source_label)
    columns[dest_index] = source_label
    columns[source_label_index] = dest_index_label


def arrange_columns(train_data_set_dataframe):
    columns = list(train_data_set_dataframe.columns)
    swap_columns(columns, 'one_rating_num', 0)
    swap_columns(columns, 'two_rating_num', 1)
    swap_columns(columns, 'three_rating_num', 2)
    swap_columns(columns, 'four_rating_num', 3)
    swap_columns(columns, 'five_rating_num', 4)
    swap_columns(columns, 'average_one_rating_phrase_percent', 5)
    swap_columns(columns, 'average_two_rating_phrase_percent', 6)
    swap_columns(columns, 'average_three_rating_phrase_percent', 7)
    swap_columns(columns, 'average_four_rating_phrase_percent', 8)
    swap_columns(columns, 'average_five_rating_phrase_percent', 9)
    swap_columns(columns, 'sentiment_phrases_found', 10)
    for i in range(0, 300):
        swap_columns(columns, 'vector_embedding_comp_' + str(i), i + 11)
    swap_columns(columns, 'rating_label', 311)
    return train_data_set_dataframe[columns]


def write_sentiment_phrases_to_disk(sentiment_phrases):
    """
    Writes the sentiment phrases computed based on the training data set in a formatted fashion.
    :param sentiment_phrases: the sentiment phrases computed based on the training set.
    """
    with open(SENTIMENT_PHRASES_FILE_NAME, 'w') as output_file:
        json.dump(sentiment_phrases, output_file)


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


def select_top_records(input_dictionary, top_percentage_to_use):
    """
    Receives a dictionary of records of the form {key: integer}, sorts the dictionary by value
    in descending order, removes records whose value is below the (1-top_percentage_to_use)
    percentile, and returns the resulting dictionary.
    :param input_dictionary: a dictionary as described above.
    :param top_percentage_to_use: the top percentage to use to filter records.
                                  Only records whose value is below no more than the input
                                  percentage of all records will be kept.
    :return: the dictionary transformed as described above.
    """
    number_of_phrases_to_include = math.floor(len(input_dictionary.keys())
                                              *
                                              min(top_percentage_to_use, 1))
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
    for phrase in sentiment_phrases_index.keys():
        if phrase in review_concatenation:
            bv[sentiment_phrases_index[phrase]] = 1
    return hash(str(bv))


def create_sentiment_phrases_index_map(sentiment_phrases):
    key_list = list(sentiment_phrases.keys())
    return {key_list[i]: i for i in range(0, len(key_list))}


def construct_data_set(dataset_name, data_set_list_of_dicts):
    counter = 0
    sentiment_phrases = dict()
    should_compute_sentiment_phrases = (dataset_name == 'train') \
                                       and \
                                       (not os.path.exists(SENTIMENT_PHRASES_FILE_NAME))
    if not should_compute_sentiment_phrases:
        print('{}: Start of loading of sentiment phrases.'.format(datetime.now()))
        with open(SENTIMENT_PHRASES_FILE_NAME, 'r') as input_file:
            sentiment_phrases = json.load(input_file)
        print('{}: End of loading of sentiment phrases.'.format(datetime.now()))
    elif dataset_name == 'train':
        print('{}: Start of computation of sentiment phrases.'.format(datetime.now()))
        sentiment_phrases = compute_sentiment_phrases(data_set_list_of_dicts)
        write_sentiment_phrases_to_disk(sentiment_phrases)
        print('{}: End of computation of sentiment phrases.'.format(datetime.now()))
    sentiment_phrases_index_map = create_sentiment_phrases_index_map(sentiment_phrases)

    for movie in data_set_list_of_dicts:
        sentiment_scores = [float(sentiment_utils.get_sentiment_score(review['text'], 1, 5)[0])
                            for review in movie['reviews']]
        movie['one_rating_num'] = len([score for score in sentiment_scores if score == 1])
        movie['two_rating_num'] = len([score for score in sentiment_scores if score == 2])
        movie['three_rating_num'] = len([score for score in sentiment_scores if score == 3])
        movie['four_rating_num'] = len([score for score in sentiment_scores if score == 4])
        movie['five_rating_num'] = len([score for score in sentiment_scores if score == 5])
        movie['average_one_rating_phrase_percent'] = sentiment_scores[1] / len(movie['reviews'])
        movie['average_two_rating_phrase_percent'] = sentiment_scores[2] / len(movie['reviews'])
        movie['average_three_rating_phrase_percent'] = sentiment_scores[3] / len(movie['reviews'])
        movie['average_four_rating_phrase_percent'] = sentiment_scores[4] / len(movie['reviews'])
        movie['average_five_rating_phrase_percent'] = sentiment_scores[5] / len(movie['reviews'])
        review_concatenation = "\n".join([review['text'] for review in movie['reviews']])
        reviews_average_vector_embedding = nlp(review_concatenation).vector
        # reviews_average_vector_embedding = nlp(review_concatenation).vector / len(movie['reviews'])
        for i in range(0, 299):
            movie['vector_embedding_comp_' + str(i)] = float(reviews_average_vector_embedding[i])
        movie['sentiment_phrases_found'] = get_sentiment_phrases_found(sentiment_phrases_index_map,
                                                                       review_concatenation)
        movie['rating_label'] = round(float(numpy.mean([review['rating']
                                            for review in movie['reviews']])))
        counter += 1
        print("{}: Dataset \'{}\': Number of movies processed: {}".format(datetime.now(),
                                                                          dataset_name, counter))
    sentiment_phrases = select_top_records(sentiment_phrases, 0.1)
    print("Number of sentiment phrases: {}".format(len(sentiment_phrases)))
    with open('{}.json'.format(dataset_name), 'w') as output_file:
        json.dump(data_set_list_of_dicts, output_file)
    return arrange_columns(pd.DataFrame(data_set_list_of_dicts))


def fill_values(dataframe, column_name, column_values):
    dataframe[column_name] = column_values
    return dataframe


def dataset_exists(dataset_name):
    return os.path.exists("{}.json".format(dataset_name))


def get_data_set(dataset_name):
    with open('{}.json'.format(dataset_name), 'r') as input_file:
        dataset = arrange_columns(pd.DataFrame(json.load(input_file)))
    return dataset


nlp = spacy.load('en_core_web_lg')

USAGE_STRING = 'Usage: sentiment_analysis.py [corpus_path] True/False\n' \
               + \
               'corpus_path: the path of the complete corpus/corpus divided into train, gold sets.\n' \
               + \
               'True/False: whether to use existing train, gold sets on disk if they exist or create ' \
               + \
               'them from scratch.'

if len(sys.argv) != 3:
    print(USAGE_STRING)
    exit(1)
if not os.path.exists(sys.argv[1]):
    print('File with path {} does not exist.'.format(sys.argv[1]))
    exit(1)
if sys.argv[2].lower() not in ['true', 'false']:
    print(USAGE_STRING)

if sys.argv[2].lower() == 'true':
    use_existing = True
else:
    use_existing = False

data_sets = dataset_utils.build_data_sets_from_json_file(sys.argv[1])

# Start of Random Forest Classification Template Code
if use_existing and dataset_exists('train'):
    train_data_set = get_data_set('train')
else:
    train_data_set = construct_data_set('train', data_sets['train'])
X_train = pd.DataFrame(train_data_set.iloc[:, 0:311].values)
y_train = train_data_set.iloc[:, 311].values

if use_existing and dataset_exists('gold'):
    gold_data_set = get_data_set('gold')
else:
    gold_data_set = construct_data_set('gold', data_sets['gold'])
X_test = pd.DataFrame(gold_data_set.iloc[:, 0:311].values)
y_ground_truth = gold_data_set.iloc[:, 311].values

# Fitting Random Forest Classification to the Training set
print("{}: Start of classification".format(datetime.now()))
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
                                    verbose=3, n_jobs=-1)
classifier.fit(X_train, y_train)
print("{}: End of classification".format(datetime.now()))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

dataset_utils.evaluate_predicted_sentiment("sentiment_analysis.eval",
                                           fill_values(gold_data_set.copy().drop('rating_label', axis=1),
                                                       'average_rating', y_pred).to_dict('records'),
                                           gold_data_set.to_dict('records'),
                                           1, 3)
