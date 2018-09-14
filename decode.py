from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import argparse
import json
import math
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

import pandas as pd
from BitVector import BitVector
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense

from utils import dataset_utils

SENTIMENT_PHRASES_FILE_NAME = 'train_sentiment_phrases.json'


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


def dataset_exists(dataset_name):
    return os.path.exists("{}.json".format(dataset_name))


def get_data_set(dataset_name):
    with open('{}.json'.format(dataset_name), 'r') as input_file:
        dataset = arrange_columns(pd.DataFrame(json.load(input_file)))
    return dataset


def get_columns_to_use(input_possible_feature_columns, input_column_dictionary, args):
    curr_columns_to_use = input_possible_feature_columns[:]
    if not args.movie_rating_distribution:
        curr_columns_to_use = [column_name for column_name in curr_columns_to_use
                               if column_name not in input_column_dictionary['movie_rating_distribution']]
    if not args.review_rating_distribution:
        curr_columns_to_use = [column_name for column_name in curr_columns_to_use
                               if column_name not in input_column_dictionary['review_rating_distribution']]
    if not args.sentiment_phrases:
        curr_columns_to_use = [column_name for column_name in curr_columns_to_use
                               if column_name not in input_column_dictionary['sentiment_phrases']]
    if not args.word_embeddings:
        curr_columns_to_use = [column_name for column_name in curr_columns_to_use
                               if column_name not in input_column_dictionary['word_embeddings']]
    if len(curr_columns_to_use) == 0: # No options selected, use all columns by default.
        curr_columns_to_use = input_possible_feature_columns
    return curr_columns_to_use


def get_sentence_data_list(input_dataset):
    sentence_data_list = []
    for movie in input_dataset:
        for sentence_data in movie['sentences_data']:
            curr_sentence_data = dict()
            curr_sentence_data['text'] = sentence_data[0]
            curr_sentence_data['id'] = movie['id']
            curr_sentence_data['sentence_sentiment'] = sentence_data[1]
            for i in range(0, len(sentence_data[2])):
                curr_sentence_data['vector_embedding_comp_' + str(i)] = sentence_data[2][i]
            curr_sentence_data['rouge_score'] = sentence_data[3]
            sentence_data_list.append(curr_sentence_data)
    return sentence_data_list


def generate_summaries_from_ranked_sentences(test_sentences_df, y_pred, number_of_sentences_in_summary):
    """
    Generates a summary for each example in the test set based on the input sentences in the same
    example's (movie's) reviews, the predicted ROUGE score for each sentence, and the desired maximum
    number of sentences in each summary.

    :param test_sentences_df: the sentences of the test set's reviews, with associated metadata.
    :param y_pred: the predicted ROUGE scores for each sentence in the test set's reviews.
    :param number_of_sentences_in_summary: the number of sentences to include in the generated summary,
                                           from the sentences of the reviews of the same respective movie.
    :return: a dictionary mapping between the id of each movie and its generated summary.
    """
    movie_to_sentences_dict = dict()
    test_sentences_df_with_pred = pd.concat(test_sentences_df, y_pred)
    test_sentences_df_with_pred.sort_values(str(len(test_sentences_df_with_pred.columns)), ascending=False)
    movie_id_column = list(test_sentences_df_with_pred.filter(items=['id'], axis=1))
    sentence_text_column = list(test_sentences_df_with_pred.filter(items=['text'], axis=1))
    for i in range(0, len(movie_id_column)):
        if not movie_id_column[i] in movie_to_sentences_dict.keys():
            movie_to_sentences_dict[movie_id_column[i]] = []
        if len(movie_to_sentences_dict[movie_id_column[i]]) < number_of_sentences_in_summary:
            movie_to_sentences_dict[movie_id_column[i]].append(sentence_text_column[i])
    return {movie_id: ". ".join(movie_to_sentences_dict[movie_id]) for movie_id in movie_to_sentences_dict.keys()}


def sort_summaries_by_input_movie_id_order(movie_id_column, summaries):
    """
    Receives a column of the ID's of movies in a dataset and a dictionary mapping between the ID of each movie
    and the respective automatically generated summary.
    The function sorts the summaries according to the same order of the ID's of the input column and returns
    a list of the sorted summaries.

    :param movie_id_column: a column of the ID's of movies in a dataset.
    :param summaries: a dictionary mapping between the ID of each movie and the respective
                      automatically generated summary.
    :return: a list of summaries sorted by the order of ID's in the input column.
    """
    sorted_summaries = []
    for movie_id in movie_id_column:
        for movie_id_key in summaries.keys():
            if movie_id == movie_id_key:
                sorted_summaries.append(summaries[movie_id_key])
    return sorted_summaries


def create_summaries(movie_id_column,
                     input_training_set, input_gold_set,
                     use_sentence_sentiment_score, use_word_embeddings,
                     number_of_sentences_in_summary):
    # If no flags for summary creation have been set, return a list of dummy, empty summaries.
    if use_sentence_sentiment_score == False and use_word_embeddings == False:
        return [''] * len(input_gold_set)
    train_sentences_data = get_sentence_data_list(input_training_set)
    gold_sentences_data = get_sentence_data_list(input_gold_set)
    train_sentences_data_df = pd.DataFrame(train_sentences_data)
    gold_sentences_data_df = pd.DataFrame(gold_sentences_data)

    possible_sentence_feature_columns = ['sentence_sentiment']
    possible_sentence_feature_columns.extend([''.join(['vector_embedding_comp_', str(i)])
                                    for i in range(0, 300)])

    sentence_data_columns = possible_sentence_feature_columns[:]
    if not use_sentence_sentiment_score:
        sentence_data_columns.remove('sentence_sentiment')
    if not use_word_embeddings:
        for i in range(0, 300):
            sentence_data_columns.remove(''.join(['vector_embedding_comp_', str(i)]))

    sentence_X_train = train_sentences_data_df.filter(items=sentence_data_columns, axis=1)
    sentence_y_train = train_sentences_data_df.loc[:, 'rouge_score'].values

    sentence_X_test = gold_sentences_data_df.filter(items=sentence_data_columns, axis=1)

    sentence_sc_X = StandardScaler()
    sentence_X_train = sentence_sc_X.fit_transform(sentence_X_train)
    sentence_X_test = sentence_sc_X.transform(sentence_X_test)

    sentence_rouge_score_regressor = Sequential()

    # Adding the input layer
    sentence_rouge_score_regressor.add(Dense(activation='relu', input_dim=len(sentence_data_columns), units=len(sentence_data_columns),
                                  kernel_initializer='uniform'))

    # Adding the first hidden layer
    sentence_rouge_score_regressor.add(Dense(activation='relu', units=len(sentence_data_columns), kernel_initializer='uniform'))
    # Adding the second hidden layer
    sentence_rouge_score_regressor.add(Dense(activation='relu', units=len(sentence_data_columns), kernel_initializer='uniform'))
    # Adding the third hidden layer
    sentence_rouge_score_regressor.add(Dense(activation='relu', units=len(sentence_data_columns), kernel_initializer='uniform'))

    # Adding the output layer
    sentence_rouge_score_regressor.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

    # Compiling the ANN
    sentence_rouge_score_regressor.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

    print("{}: Start of ranking of sentences".format(datetime.now()))

    # Fitting the ANN to the Training set
    sentence_rouge_score_regressor.fit(sentence_X_train, sentence_y_train, batch_size=10, epochs=10)
    sentence_y_pred = classifier.predict(sentence_X_test)

    print("{}: End of ranking of sentences".format(datetime.now()))

    print("{}: Start of creation of summaries".format(datetime.now()))

    summaries = generate_summaries_from_ranked_sentences(gold_sentences_data_df.drop('rouge_score', axis=1),
                                                         sentence_y_pred,
                                                         number_of_sentences_in_summary)
    summaries = sort_summaries_by_input_movie_id_order(movie_id_column, summaries)

    print("{}: Start of creation of summaries".format(datetime.now()))

    return summaries


def create_decoded_items_list(id_column, y_sentiment_pred, y_sentiment_ground_truth,
                              y_summary_pred, y_summary_ground_truth):
    """
    Receives lists of predicted sentiment, ground truth sentiment, predicted summaries and
    ground truth summaries, and concatenates them together into a list of dictionaries, where
    a single dictionary exists for each example in the test set.
    :param id_column: the id of the example.
    :param y_sentiment_pred: a list of predicted sentiment values.
    :param y_sentiment_ground_truth: a list of ground truth sentiment values.
    :param y_summary_pred: a list of predicted summaries.
    :param y_summary_ground_truth: a list of ground truth summaries.
    :return: a list of dictionaries as described above.
    """
    result = []
    for i in range(0, len(y_sentiment_pred)):
        result.append({'id': int(id_column[i]),
                       'predicted_sentiment': int(y_sentiment_pred[i]),
                       'ground_truth_sentiment': int(y_sentiment_ground_truth[i]),
                       'predicted_summary': y_summary_pred[i],
                       'ground_truth_summary': y_summary_ground_truth[i]})
    return result


parser = argparse.ArgumentParser(description='Receives a training set and a gold set which is '
                                             +
                                             'considered to be identical to the test set, except '
                                             +
                                             'with an additional ground truth label column.\n'
                                             +
                                             'The decode script trains a model with the input '
                                             +
                                             'data from the training set, performs sentiment analysis '
                                             +
                                             'and textual summarization and and produces as a result '
                                             +
                                             'a file called decoded.json, containing for each example '
                                             +
                                             'in the test set, its predicted sentiment and produced '
                                             +
                                             'textual summary.\n'
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
parser.add_argument("TRAIN_PATH",
                    help="The path of training set.")
parser.add_argument("GOLD_PATH",
                    help="The path of the gold set (will also be used as a test set).")
parser.add_argument("OUTPUT_FILE_PATH",
                    help="The path in which to save the output file, including the file's name.")
parser.add_argument("NUM_OF_SENTENCES_IN_SUMMARY",
                    type=int,
                    help="The number of sentences to include in the automatically generated summary of "
                         "each movie.")
parser.add_argument("-mrd", "--movie_rating_distribution", action="store_true",
                    help="Use the distribution of 1,2,3,4,5 ratings per movie.")
parser.add_argument("-rrd", "--review_rating_distribution", action="store_true",
                    help="Use the average percentages of 1,2,3,4,5 ratings per review.")
parser.add_argument("-we", "--word_embeddings", action="store_true",
                    help="Use word embeddings computed based on word2vec.")
parser.add_argument("-sp", "--sentiment_phrases", action="store_true",
                    help="Use the presence or lack of presence of identified "
                         +
                         "sentiment carrying phrases.")
parser.add_argument("-ts_ss", "--textual_summarization_sentence_sentiment", action="store_true",
                    help="Use the sentiment score of each sentence to rank it for deciding whether to "
                         +
                         "choose to include it in a textual summary.")
parser.add_argument("-ts_swe", "--textual_summarization_sentence_word_embeddings", action="store_true",
                    help="Use the sum of word embeddings of each sentence to rank the sentence to decide whether "
                         +
                         "to choose to include it in a textual summary.")

args = parser.parse_args()

if not os.path.exists(args.TRAIN_PATH):
    print('Error: the file {} does not exist.'.format(args.CORPUS_PATH))
    exit(1)
elif not os.path.exists(args.GOLD_PATH):
    print('Error: the file {} does not exist.'.format(args.CORPUS_PATH))
    exit(1)
elif not os.path.exists(os.path.dirname(args.OUTPUT_FILE_PATH)):
    print('Error: the directory {} does not exist.'.format(os.path.dirname(args.OUTPUT_FILE_PATH)))
    exit(1)

with open(args.TRAIN_PATH, 'r') as train_file:
    train_json_object = json.load(train_file)
    train_data_set = pd.DataFrame(train_json_object)
with open(args.GOLD_PATH, 'r') as gold_file:
    gold_json_object = json.load(gold_file)
    gold_data_set = pd.DataFrame(gold_json_object)

movie_rating_distribution_columns = ['one_rating_num', 'two_rating_num', 'three_rating_num',
                                     'four_rating_num', 'five_rating_num']
review_rating_distribution_columns = ['average_one_rating_phrase_percent',
                                      'average_two_rating_phrase_percent',
                                      'average_three_rating_phrase_percent',
                                      'average_four_rating_phrase_percent',
                                      'average_five_rating_phrase_percent']
sentiment_phrases_columns = ['sentiment_phrases_found']
word_embedding_columns = ['vector_embedding_comp_' + str(i) for i in range(0, 300)]
column_dictionary = {'movie_rating_distribution': movie_rating_distribution_columns,
                     'review_rating_distribution': review_rating_distribution_columns,
                     'sentiment_phrases': sentiment_phrases_columns,
                     'word_embeddings': word_embedding_columns}

possible_feature_columns = [column_name for key in column_dictionary.keys()
                            for column_name in column_dictionary[key]]

columns_to_use = get_columns_to_use(possible_feature_columns, column_dictionary, args)
print("Columns used: {}".format(columns_to_use))

X_train = train_data_set.filter(items=columns_to_use, axis=1)
y_train = train_data_set.loc[:, 'rating_label'].values
print("y_train before normalizing: {}".format(y_train))
y_train = [dataset_utils.shift_scale(old_value, 1, 5, 0, 1) for old_value in y_train]
print("y_train after normalizing: {}".format(y_train))

X_test = gold_data_set.filter(items=columns_to_use, axis=1)
y_ground_truth = gold_data_set.loc[:, 'rating_label'].values

print("{}: Start of classification".format(datetime.now()))
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
#                                     verbose=3, n_jobs=-1)
# classifier.fit(X_train, y_train)

# Predicting sentiment scores

classifier = Sequential()

# Adding the input layer
classifier.add(Dense(activation='relu', input_dim=len(columns_to_use), units=len(columns_to_use),
                     kernel_initializer='uniform'))

# Adding the first hidden layer
classifier.add(Dense(activation='relu', units=len(columns_to_use), kernel_initializer='uniform'))
# Adding the second hidden layer
classifier.add(Dense(activation='relu', units=len(columns_to_use), kernel_initializer='uniform'))
# Adding the third hidden layer
classifier.add(Dense(activation='relu', units=len(columns_to_use), kernel_initializer='uniform'))

# Adding the output layer
classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=10)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("y_pred before normalizing: {}".format(y_pred))
y_pred = [round(float(dataset_utils.shift_scale(old_value, 0, 1, 1, 5))) for old_value in y_pred]
print("y_pred after normalizing: {}".format(y_pred))
print("{}: End of classification".format(datetime.now()))

# Creating summaries.
print("{}: Start of creation of summaries".format(datetime.now()))
y_summary = create_summaries(gold_data_set.filter(items=['id'], axis=1).iloc[:, 0],
                             train_json_object,
                             gold_json_object,
                             args.textual_summarization_sentence_sentiment,
                             args.textual_summarization_sentence_word_embeddings,
                             args.NUM_OF_SENTENCES_IN_SUMMARY)
print("{}: End of creation of summaries".format(datetime.now()))

print("{}: Start of creation of decoded examples' file".format(datetime.now()))
with open(args.OUTPUT_FILE_PATH, 'w') as output_file:
    json.dump(create_decoded_items_list(gold_data_set.loc[:, 'id'],
                                        y_pred, y_ground_truth, y_summary,
                                        gold_data_set.loc[:, 'summary']),
              output_file)
print("{}: End of creation of decoded examples' file".format(datetime.now()))
