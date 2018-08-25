#!/usr/bin/python
import sys
import os

from dataset_utils import dataset_utils
from sumeval.metrics.rouge import RougeCalculator
from textblob import TextBlob
import datetime

USAGE_STRING = 'Usage: python baseline_model.py [corpus_path] True/False'
NORMALIZED_RANGE_MIN = 1
NORMALIZED_RANGE_MAX = 5


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
        review_ratings = [shift_scale(review['rating'], 1, 5, NORMALIZED_RANGE_MIN, NORMALIZED_RANGE_MAX)
                          for review in movie['reviews']]
        movie['rating_label'] = sum(review_ratings) / len(review_ratings)
        for review in movie['reviews']:
            sentiment_score += get_sentiment_score(review['text'])

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
            sentiment_score += get_sentiment_score(review['text'])
        sentiment_score_average = sentiment_score / len(movie['reviews'])

        movie['average_rating'] = min(round(sentiment_score_average * sentiment_ratio), NORMALIZED_RANGE_MAX)

        # Set the first summary as the baseline summary:
        # we take the first 5 words of a random review as a summary.
        first_review_words = movie['reviews'][0]['text'].split()
        first_review_length = len(first_review_words)
        movie['summary'] = movie['reviews'][0]['text'].split()[:min(5, first_review_length)]
        movie['summary'] = ' '.join(movie['summary'])
        movie_count += 1
        print("Movies decoded: {0}, computed rating before rounding: {1}, after rounding: {2}"
              .format(str(movie_count),
                      sentiment_score_average,
                      round(sentiment_score_average * sentiment_ratio)))

    return test_data


def prepare_gold_data(gold_data_set):
    # for movie in gold_data_set:
    #     overall_score = 0
    #     for review in gold_data_set[movie]['data']:
    #         overall_score += review['overall']
    #     overall_score_average = round(overall_score / gold_data_set[product]['data'].__len__())
    #     gold_data_set[movie]['overall'] = overall_score_average
    #
    #     # set a the first summary as the baseline summary
    #     gold_data_set[product]['summary'] = gold_data_set[product]['data'][0]['summary']

    return gold_data_set


def shift_scale(old_value, old_min, old_max, new_min, new_max):
    return ((new_max - new_min) / (old_max - old_min)) * (old_value - old_max) + new_max;


def get_sentiment_score(text):
    """
    Normalizes the sentiment score from a scale of [-1, 1] to the chosen normalized
    discrete scale of {1,2,3,4,5}.
    :param text: the input text whose sentiment score is to be computed and returned.
    :return: the sentiment score of the input text, normalized to a discrete scale of {1,2,3,4,5}.
    """
    return round(shift_scale(TextBlob(text).sentiment.polarity, -1, 1, NORMALIZED_RANGE_MIN, NORMALIZED_RANGE_MAX))


def evaluate_predicted_sentiment(decoded_test_data, gold_data):
    eval_file = open('baselineOverAllEvaluation', 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Over All Evaluation - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\t\t\taccuracy\n'])
    counter = 0
    accuracy_percentage_sum = 0
    for decoded_movie in decoded_test_data:
        for gold_movie in gold_data:
            if decoded_movie['id'] == gold_movie['id']:
                review_ratings = [shift_scale(review['rating'], 1, 5, NORMALIZED_RANGE_MIN, NORMALIZED_RANGE_MAX)
                                  for review in gold_movie['reviews']]
                gold_movie['rating_label'] = round(sum(review_ratings) / len(review_ratings))
                # print('test rating: {}, gold rating: {}'.format(decoded_movie['average_rating'],
                #                                                 gold_movie['rating_label']))
                if decoded_movie['average_rating'] == gold_movie['rating_label']:
                    accuracy_percentage_sum += 1
        counter += 1
        accuracy_percentage = (accuracy_percentage_sum * 1.0) / counter
        eval_file.write(str(counter) + '\t\t\t' + str(accuracy_percentage) + '\n')
    eval_file.write('# ------------------------\n')
    eval_file.write('Overall Average Accuracy:' + str(accuracy_percentage))
    eval_file.close()


def evaluate_summary(decoded_test_data, gold_data, n_gram_order):
    eval_file = open('baseline summary evaluation-ROUGE_' + str(n_gram_order), 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Summarization - Rouge_', str(n_gram_order), ' - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\t\t\tRecall\t\t\tPrecision\t\t\tFscore\n'])
    counter = 0
    recall_sum = 0
    precision_sum = 0
    f_score_sum = 0
    test_to_gold_map = dict()
    for decoded_movie in decoded_test_data:
        for gold_movie in gold_data:
            if decoded_movie['id'] == gold_movie['id']:
                test_to_gold_map[decoded_movie['id']] = (decoded_movie, gold_movie)

    for decoded_movie_id in test_to_gold_map.keys():
        rouge = calculate_rouge(test_to_gold_map[decoded_movie_id][1]["summary"],
                                test_to_gold_map[decoded_movie_id][0]["summary"],
                                n_gram_order)
        recall_sum += rouge['recall']
        precision_sum += rouge['precision']
        f_score_sum += rouge['fScore']

        eval_file.write(
            str(counter) + '\t\t\t' + str(rouge['recall']) + '\t\t\t' + str(rouge['precision']) + '\t\t\t' + str(
                rouge['fScore']) + '\n')

    eval_file.write('# ------------------------\n')
    eval_file.write('Average Recall:' + str(recall_sum / len(decoded_test_data)) + '\n')
    eval_file.write('Average Gold Precision:' + str(precision_sum / len(decoded_test_data)) + '\n')
    eval_file.write('Average Gold FScore:' + str(f_score_sum / len(decoded_test_data)) + '\n')
    eval_file.close()


def calculate_rouge(summary_gold, summary_test, ngram_order):
    rouge = RougeCalculator(stopwords=True, lang="en")
    rouge_recall = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0.5)

    rouge_precision = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0.5)

    rouge_f_score = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0.5)
    # rouge_recall = rouge.rouge_n(
    #     summary=summary_test,
    #     references=[summary_gold],
    #     n=ngram_order, alpha=0)
    #
    # rouge_precision = rouge.rouge_n(
    #     summary=summary_test,
    #     references=[summary_gold],
    #     n=ngram_order, alpha=1)
    #
    # rouge_f_score = rouge.rouge_n(
    #     summary=summary_test,
    #     references=[summary_gold],
    #     n=ngram_order, alpha=0.5)
    return {'recall': rouge_recall, 'precision': rouge_precision, 'fScore': rouge_f_score}


def main():
    # Usage of the program is of the form 'python baseline_model.py [dataset_path]'.
    if len(sys.argv) > 3:
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
    evaluate_predicted_sentiment(decoded_test_data, gold_data)
    evaluate_summary(decoded_test_data, gold_data, 1)
    evaluate_summary(decoded_test_data, gold_data, 2)
    print("{} : End of evaluation".format(datetime.datetime.now()))


main()
