#!/usr/bin/python
import sys
import os
from dataset_utils import dataset_utils
from sumeval.metrics.rouge import RougeCalculator
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import datetime

USAGE_STRING = 'Usage: python baseline_model.py [corpus_path]'

nltk.download('stopwords')

english_negation_words = {'no', 'not', 'nothing', 'never', 'nowhere'}
stopwords_to_omit = set(stopwords.words('english')).union(english_negation_words)
ps = PorterStemmer()

def train_model(train_data_set):
    """
    Trains the baseline model by computing the average sentiment score
    for every movie and computing the ratio between the average sentiment
    score and the average rating given by critics.
    :param train_data_set: Training set used for training.
    :return: Training set with additional required computed data
             after training.
    """
    for movie in train_data_set:
        sentiment_score = 0
        # Normalize the average rating given by critics to the movie
        # from a scale of 0 to 10 to a scale of 1 to 5.
        real_average_rating = round(float(movie['average_rating'].strip()) / 2)
        movie['rating_label'] = real_average_rating
        for review in movie['reviews']:
            sentiment_score += get_sentiment_score(review['text'])

        sentiment_score_average = round(sentiment_score / len(movie['reviews']))
        movie['sentiScore'] = sentiment_score_average

        # Set the ratio between the "real" rating of the movie and the average of the sentiment scores
        # computed for each review, averaged over the number of reviews.
        movie['sentiRatio'] = real_average_rating / sentiment_score_average

    return train_data_set


def decode(test_data, trained_data):
    sentiment_score = 0

    test_to_trained_map = dict()

    for test_movie in test_data:
        for trained_movie in trained_data:
            test_to_trained_map[test_movie] = trained_movie

    for movie in test_data:
        for review in movie['reviews']:
            sentiment_score += get_sentiment_score(review['text'])
        sentiment_score_average = sentiment_score / len(movie['reviews'])
        movie['average_rating'] = round(sentiment_score_average * test_to_trained_map[movie]['sentiRatio'])
        # if test_data[movie]['overall'] > 5:
        #     test_data[movie]['overall'] = 5

        # Set the first summary as the baseline summary:
        # we take the first 5 words of a random review as a summary.
        first_review_words = movie['reviews'][0]['text'].split()
        first_review_length = len(first_review_words)
        movie['summary'] = movie['reviews'][0]['text'].split()[:min(5, first_review_length)]
        test_data[movie]['summary'] = ' '.join(test_data[movie]['summary'])

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
    return round(shift_scale(TextBlob(text).sentiment.polarity, -1, 1, 1, 5))


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
            # accuracy_percentage = ((abs(gold_data[movie]['average_rating']
            #                         -
            #                         abs(decoded_test_data[movie]['average_rating']
            #                         -
            #                         gold_data[movie]['average_rating']))) / gold_data[movie]['average_rating'])\
            #                         * \
            #                         100
            if decoded_movie['id'] == gold_movie['id'] \
                    and decoded_movie['average_rating'] == gold_movie['average_rating']:
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
            test_to_gold_map[decoded_movie] = gold_movie

    for decoded_movie in test_to_gold_map.keys():
        rouge = calculate_rouge(test_to_gold_map[decoded_movie]["summary"],
                                decoded_movie["summary"],
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
        n=ngram_order, alpha=0)

    rouge_precision = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0)

    rouge_f_score = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0)
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


def preprocess_text(text):
    """
    Preprocesses the input text and returns the preprocessed string.
    Preprocessing includes removing numbers, converting all words to lowercase, removing stopwords and
    stemming each remaining word.
    :param text: the input text to preprocess.
    :return: the preprocessed string.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords_to_omit]
    text = ' '.join(text)
    return text


def preprocess_datasets(data_sets):
    movie_count = 0
    for key in data_sets.keys():
        for movie in data_sets[key]:
            movie['summary'] = preprocess_text(movie['summary'])
            for review in movie['reviews']:
                review['text'] = preprocess_text(review['text'])
        movie_count += 1
        print(movie_count)
    return data_sets


def main():
    # Usage of the program is of the form 'python baseline_model.py [dataset_path]'.
    if (len(sys.argv) > 2) or not os.path.exists(sys.argv[1]):
        print(USAGE_STRING)
        exit(1)

    data_sets = dataset_utils.build_data_sets_from_json_file(sys.argv[1])
    print("{} : Starting preprocessing".format(datetime.datetime.now()))
    data_sets = preprocess_datasets(data_sets)
    print("{} : End of preprocessing".format(datetime.datetime.now()))
    trained_data = train_model(data_sets['train'])
    decoded_test_data = decode(data_sets['test'], trained_data)
    gold_data = prepare_gold_data(data_sets['gold'])
    evaluate_predicted_sentiment(decoded_test_data, gold_data)
    evaluate_summary(decoded_test_data, gold_data, 1)
    evaluate_summary(decoded_test_data, gold_data, 2)


main()
