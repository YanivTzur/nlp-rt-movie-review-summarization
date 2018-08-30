import datetime
import os
import sys

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


def construct_data_set(dataset_name, data_set_list_of_dicts):
    counter = 0
    for movie in data_set_list_of_dicts:
        # sentiment_scores = [sentiment_utils.get_sentiment_score(review['text'], 1, 5)[0]
        #                     for review in movie['reviews']]
        # movie['one_rating_num'] = len([score for score in sentiment_scores if score == 1])
        # movie['two_rating_num'] = len([score for score in sentiment_scores if score == 2])
        # movie['three_rating_num'] = len([score for score in sentiment_scores if score == 3])
        # movie['four_rating_num'] = len([score for score in sentiment_scores if score == 4])
        # movie['five_rating_num'] = len([score for score in sentiment_scores if score == 5])
        # movie['average_one_rating_phrase_percent'] = sentiment_scores[1] / len(movie['reviews'])
        # movie['average_two_rating_phrase_percent'] = sentiment_scores[2] / len(movie['reviews'])
        # movie['average_three_rating_phrase_percent'] = sentiment_scores[3] / len(movie['reviews'])
        # movie['average_four_rating_phrase_percent'] = sentiment_scores[4] / len(movie['reviews'])
        # movie['average_five_rating_phrase_percent'] = sentiment_scores[5] / len(movie['reviews'])
        # # for review in movie['reviews']:
        #     reviews_average_vector_embedding += nlp(review['text']).vector / len(movie['reviews'])
        review_concatenation = "\n".join([review['text'] for review in movie['reviews']])
        reviews_average_vector_embedding = nlp(review_concatenation).vector
        # reviews_average_vector_embedding = nlp(review_concatenation).vector / len(movie['reviews'])
        for i in range(0,300):
            movie['vector_embedding_comp_' + str(i)] = reviews_average_vector_embedding[i]

        movie['rating_label'] = round(numpy.mean([review['rating'] for review in movie['reviews']]))
        # print("Dataset \'{}\': id: {}, average_rating: {}, rating_label: {}"
        #       .format(dataset_name, movie['id'], movie['average_rating'],
        #               movie['rating_label']))
        counter += 1
        print("Dataset \'{}\': Number of movies processed: {}".format(dataset_name, counter))
    train_data_set_dataframe = pd.DataFrame(data_set_list_of_dicts)
    columns = list(train_data_set_dataframe.columns)
    # swap_columns(columns, 'one_rating_num', 0)
    # swap_columns(columns, 'two_rating_num', 1)
    # swap_columns(columns, 'three_rating_num', 2)
    # swap_columns(columns, 'four_rating_num', 3)
    # swap_columns(columns, 'five_rating_num', 4)
    # swap_columns(columns, 'average_one_rating_phrase_percent', 5)
    # swap_columns(columns, 'average_two_rating_phrase_percent', 6)
    # swap_columns(columns, 'average_three_rating_phrase_percent', 7)
    # swap_columns(columns, 'average_four_rating_phrase_percent', 8)
    # swap_columns(columns, 'average_five_rating_phrase_percent', 9)
    for i in range(0, 300):
        swap_columns(columns, 'vector_embedding_comp_' + str(i), i)
    swap_columns(columns, 'rating_label', 300)
    train_data_set_dataframe = train_data_set_dataframe[columns]
    return train_data_set_dataframe


def fill_values(dataframe, column_name, column_values):
    dataframe[column_name] = column_values
    return dataframe


nlp = spacy.load('en_core_web_lg')

if len(sys.argv) != 2:
    print('Usage: sentiment_analysis.py [corpus_path]')
    exit(1)
if not os.path.exists(sys.argv[1]):
    print('File with path {} does not exist.'.format(sys.argv[1]))
    exit(1)

data_sets = dataset_utils.build_data_sets_from_json_file(sys.argv[1])

# Start of Random Forest Classification Template Code

train_data_set = construct_data_set('train', data_sets['train'])
X_train = pd.DataFrame(train_data_set.iloc[:, 0:299].values)
y_train = train_data_set.iloc[:, 300].values

gold_data_set = construct_data_set('gold', data_sets['gold'])
X_test = pd.DataFrame(gold_data_set.iloc[:, 0:299].values)
y_ground_truth = gold_data_set.iloc[:, 300].values

# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
#
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
print("{}: Start of classification".format(datetime.datetime.now()))
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0,
                                    verbose=3, n_jobs=-1)
classifier.fit(X_train, y_train)
print("{}: End of classification".format(datetime.datetime.now()))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

dataset_utils.evaluate_predicted_sentiment(fill_values(gold_data_set.copy().drop('rating_label', axis=1),
                                                       'average_rating', y_pred).to_dict('records'),
                                           gold_data_set.to_dict('records'),
                                           1, 5)

# End of Random Forest Classification Template Code
