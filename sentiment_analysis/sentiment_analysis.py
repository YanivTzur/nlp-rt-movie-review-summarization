import datetime
import os
import sys
import json
from textblob import TextBlob

from utils import dataset_utils
from utils import sentiment_utils
import pandas as pd
import numpy
import spacy
from sklearn.ensemble import RandomForestClassifier


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
    for i in range(0, 300):
        swap_columns(columns, 'vector_embedding_comp_' + str(i), i + 10)
    swap_columns(columns, 'rating_label', 310)
    return train_data_set_dataframe[columns]


def construct_data_set(dataset_name, data_set_list_of_dicts):
    counter = 0
    sentiment_phrases = set()

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
        sentiment_phrases += [" ".join(assesysment[0]) for assessment
                              in TextBlob(review_concatenation).sentiment_assessments.assessments]
        reviews_average_vector_embedding = nlp(review_concatenation).vector
        # reviews_average_vector_embedding = nlp(review_concatenation).vector / len(movie['reviews'])
        for i in range(0, 300):
            movie['vector_embedding_comp_' + str(i)] = float(reviews_average_vector_embedding[i])
        movie['rating_label'] = float(round(numpy.mean([review['rating'] for review in movie['reviews']])))
        counter += 1
        print("Dataset \'{}\': Number of movies processed: {}".format(dataset_name, counter))
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

USAGE_STRING = 'Usage: sentiment_analysis.py [corpus_path] True/False\n'\
               +\
               'corpus_path: the path of the complete corpus/corpus divided into train, gold sets.\n'\
               +\
               'True/False: whether to use existing train, gold sets on disk if they exist or create '\
               +\
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
X_train = pd.DataFrame(train_data_set.iloc[:, 0:310].values)
y_train = train_data_set.iloc[:, 310].values

if use_existing and dataset_exists('gold'):
    gold_data_set = get_data_set('gold')
else:
    gold_data_set = construct_data_set('gold', data_sets['gold'])
X_test = pd.DataFrame(gold_data_set.iloc[:, 0:310].values)
y_ground_truth = gold_data_set.iloc[:, 310].values

# Fitting Random Forest Classification to the Training set
print("{}: Start of classification".format(datetime.datetime.now()))
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
                                    verbose=3, n_jobs=-1)
classifier.fit(X_train, y_train)
print("{}: End of classification".format(datetime.datetime.now()))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

dataset_utils.evaluate_predicted_sentiment("sentiment_analysis.eval",
                                           fill_values(gold_data_set.copy().drop('rating_label', axis=1),
                                                       'average_rating', y_pred).to_dict('records'),
                                           gold_data_set.to_dict('records'),
                                           1, 5)
