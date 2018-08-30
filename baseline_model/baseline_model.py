#!/usr/bin/python
import sys
import os
import random
from utils import dataset_utils, sentiment_utils
import datetime

USAGE_STRING = 'Usage: python baseline_model.py [corpus_path] True/False'
NORMALIZED_RANGE_MIN = 1  # The minimum value in the set of values a sentiment score
                          # can receive.
NORMALIZED_RANGE_MAX = 5  # The maximum value in the set of values a sentiment score


# can receive.


def train_model(train_data_set):
    """
    Trains the baseline model by computing the average sentiment score
    for every movie and computing the ratio between the average sentiment
    score and the average rating given by critics.
    :param train_data_set: Training set used for training.
    :return: Training set with additional required computed data
             after training.
    """
    movie_count = 0
    total_reviews_num = len([review for movie in train_data_set for review in movie['reviews']])
    all_movies_sent_average = 0

    for movie in train_data_set:
        sentiment_score = 0
        review_ratings = [dataset_utils.shift_scale(review['rating'], 1, 5,
                                                    NORMALIZED_RANGE_MIN, NORMALIZED_RANGE_MAX)
                          for review in movie['reviews']]
        movie['rating_label'] = sum(review_ratings) / len(review_ratings)
        for review in movie['reviews']:
            sentiment_score += sentiment_utils.get_sentiment_score(review['text'], NORMALIZED_RANGE_MIN,
                                                                   NORMALIZED_RANGE_MAX)

        sentiment_score_average = sentiment_score / len(movie['reviews'])
        curr_movie_sentiment_average = movie['rating_label'] / sentiment_score_average
        all_movies_sent_average += (curr_movie_sentiment_average * len(movie['reviews'])) \
                                   / \
                                   total_reviews_num
        # Set the ratio between the "real" rating of the movie and the average of
        # the sentiment scores computed for each review, averaged over the number of reviews.
        movie_count += 1
        print("Movies trained: " + str(movie_count))

    sentiment_ratio = min(all_movies_sent_average, NORMALIZED_RANGE_MAX)
    print("{0}: Trained sentiment ratio is {1}".format(datetime.datetime.now(),
                                                       sentiment_ratio))

    return sentiment_ratio


def decode(test_data, sentiment_ratio):
    movie_count = 0

    for movie in test_data:
        sentiment_score = 0
        for review in movie['reviews']:
            sentiment_score += sentiment_utils.get_sentiment_score(review['text'],
                                                                   NORMALIZED_RANGE_MIN,
                                                                   NORMALIZED_RANGE_MAX)
        sentiment_score_average = sentiment_score / len(movie['reviews'])

        movie['average_rating'] = min(round(sentiment_score_average * sentiment_ratio), NORMALIZED_RANGE_MAX)

        # We take the first sentence of a random summary as the chosen summary.
        random.seed(0)  # Set constant seed so every user gets the same results.
        movie['summary'] = movie['reviews'][random.randint(0, len(movie['reviews']) - 1)]['text'].split('.')[0]
        movie_count += 1
        print("Movies decoded: {0}, computed rating before modification: {1}, after modification and rounding: {2}"
              .format(str(movie_count),
                      sentiment_score_average,
                      round(sentiment_score_average * sentiment_ratio)))

    return test_data


def prepare_gold_data(gold_data_set):
    return gold_data_set


def main():
    if len(sys.argv) != 3:
        print(USAGE_STRING)
        exit(1)
    if not os.path.exists(sys.argv[1]):
        print('File with path {} does not exist.'.format(sys.argv[1]))
        exit(1)
    if sys.argv[2].lower() not in ['true', 'false']:
        print(USAGE_STRING)
        exit(1)

    if sys.argv[2].lower() == 'true':
        preprocess = True
    else:
        preprocess = False
    data_sets = dataset_utils.build_data_sets_from_json_file(sys.argv[1], preprocess)
    print("{} : Starting training".format(datetime.datetime.now()))
    trained_sentiment_ratio = train_model(data_sets['train'])
    print("{} : End of training".format(datetime.datetime.now()))
    print("{} : Starting decoding".format(datetime.datetime.now()))
    decoded_test_data = decode(data_sets['test'], trained_sentiment_ratio)
    print("{} : End of decoding".format(datetime.datetime.now()))
    gold_data = prepare_gold_data(data_sets['gold'])
    print("{} : Starting evaluation".format(datetime.datetime.now()))
    dataset_utils.evaluate_predicted_sentiment(decoded_test_data, gold_data,
                                               NORMALIZED_RANGE_MIN, NORMALIZED_RANGE_MAX)
    dataset_utils.evaluate_summary(decoded_test_data, gold_data, 1)
    dataset_utils.evaluate_summary(decoded_test_data, gold_data, 2)
    print("{} : End of evaluation".format(datetime.datetime.now()))


main()
