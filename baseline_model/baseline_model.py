#!/usr/bin/python
import sys
import os
from dataset_utils import dataset_utils
from sumeval.metrics.rouge import RougeCalculator

USAGE_STRING = 'Usage: python baseline_model.py [corpus_path]'


def prepare_train_data(train_data_set):
    """
    Trains the baseline model by computing the average sentiment score
    for every movie and computing the ratio between the average sentiment
    score and the average rating given by critics.
    :param train_data_set: Training set used for training.
    :return: Training set with additional required computed data
             after training.
    """
    for product in train_data_set:
        sentiment_score = 0
        overall_score = 0
        for review in train_data_set[product]['data']:
            overall_score += review['overall']

            sentiment_score += get_sentiment_score(review['reviewText'])
        overall_score_average = overall_score / train_data_set[product]['data'].__len__()
        train_data_set[product]['overAll'] = overall_score_average

        sentiment_score_average = sentiment_score / train_data_set[product]['data'].__len__()
        train_data_set[product]['sentiScore'] = sentiment_score_average

        # set the ratio between sentiment analysis and aver all score
        train_data_set[product]['sentiRatio'] = overall_score_average / sentiment_score_average

        # #set a random sammery as the baseline summery
        # randomReview = random.randrange(0, trainDataSet[product]['data'].__len__())
        # trainDataSet[product]['summary'] = trainDataSet[product]['data'][randomReview]['summary']

    return train_data_set


def prepare_gold_data(gold_data_set):
    for product in gold_data_set:
        overall_score = 0
        for review in gold_data_set[product]['data']:
            overall_score += review['overall']
        overall_score_average = round(overall_score / gold_data_set[product]['data'].__len__())
        gold_data_set[product]['overAll'] = overall_score_average

        # set a the first summary as the baseline summary
        gold_data_set[product]['summary'] = gold_data_set[product]['data'][0]['summary']

    return gold_data_set


def decode(test_data, trained_data):
    for product in test_data:
        sentiment_score = 0
        for review in test_data[product]['data']:
            sentiment_score += get_sentiment_score(review['reviewText'])
        sentiment_score_average = sentiment_score / test_data[product]['data'].__len__()
        test_data[product]['overAll'] = round(sentiment_score_average * trained_data[product]['sentiRatio'])
        if test_data[product]['overAll'] > 5:
            test_data[product]['overAll'] = 5

        # Set the first summary as the baseline summary - we take the first 5 words of a random review as a summary.
        if test_data[product]['data'][0]['reviewText'].split().__len__() >= 5:
            test_data[product]['summary'] = test_data[product]['data'][0]['reviewText'].split()[:5]
        else:
            test_data[product]['summary'] = test_data[product]['data'][0]['reviewText'] \
                                                .split()[:test_data[product]['data'][0]['reviewText'].split().__len__()]
        test_data[product]['summary'] = ' '.join(test_data[product]['summary'])

    return test_data


def get_sentiment_score(text):
    # blob = TextBlob(text)
    sentence_count = 0
    text_sentiment_count = 0
    for sentence in blob.sentences:
        sentence_count += 1
        text_sentiment_count += sentence.sentiment.polarity
    if text_sentiment_count == 0:
        return 0
    text_sentiment_average = text_sentiment_count / sentence_count
    # shift the sentiment score from -1 - 1 to 0 -5 like the overall score
    sentiment_score = round(((text_sentiment_average - (-1)) * (5 - 0)) / (1 - (-1))) + 0
    return sentiment_score


def evaluate_predicted_sentiment(decoded_test_data, gold_data):
    eval_file = open('baselineOverAllEvaluation', 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Over All Evaluation - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\t\t\taccuracy\n'])
    counter = 0
    accuracy_percentage_sum = 0
    for product in decoded_test_data:
        accuracy_percentage = ((abs(gold_data[product]['overAll']
                                    - abs(decoded_test_data[product]['overAll']
                                          - gold_data[product]['overAll']))) / gold_data[product]['overAll']) * 100
        if accuracy_percentage == 100.0:
            accuracy_percentage_sum += 1
        counter += 1
        eval_file.write(str(counter) + '\t\t\t' + str(accuracy_percentage) + '\n')
    eval_file.write('# ------------------------\n')
    eval_file.write('Over All Average Accuracy:' + str(accuracy_percentage_sum / decoded_test_data.__len__()))
    eval_file.close()


def evaluate_summary(decoded_test_data, gold_data, n_gram_order):
    eval_file = open('baselineESummarevaluation-ROUGE_' + str(n_gram_order), 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Summerazion - Rouge_', str(n_gram_order), ' - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\t\t\tRecall\t\t\tPrecision\t\t\tFscore\n'])
    counter = 0
    recall_count = 0
    precision_count = 0
    f_score_count = 0

    for product in decoded_test_data:
        rouge = calculate_rouge(gold_data[product]["summary"], decoded_test_data[product]["summary"], n_gram_order)
        recall_count += rouge['recall']
        precision_count += rouge['precision']
        f_score_count += rouge['fScore']

        eval_file.write(
            str(counter) + '\t\t\t' + str(rouge['recall']) + '\t\t\t' + str(rouge['precision']) + '\t\t\t' + str(
                rouge['fScore']) + '\n')

    eval_file.write('# ------------------------\n')
    eval_file.write('Average Recall:' + str(recall_count / decoded_test_data.__len__()) + '\n')
    eval_file.write('Average Gold Precision:' + str(precision_count / decoded_test_data.__len__()) + '\n')
    eval_file.write('Average Gold FScore:' + str(f_score_count / decoded_test_data.__len__()) + '\n')
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
        n=ngram_order, alpha=1)

    rouge_f_score = rouge.rouge_n(
        summary=summary_test,
        references=[summary_gold],
        n=ngram_order, alpha=0.5)
    return {'recall': rouge_recall, 'precision': rouge_precision, 'fScore': rouge_f_score}


def main():
    # Usage of the program is of the form 'python baseline_model.py [dataset_path]'.
    if (len(sys.argv) > 2) or not os.path.exists(sys.argv[1]):
        print(USAGE_STRING)
        exit(1)

    data_sets = dataset_utils.build_data_sets_from_json_file(sys.argv[1])
    trained_data = prepare_train_data(data_sets['train'])
    decoded_test_data = decode(data_sets['test'], trained_data)
    gold_data = prepare_gold_data(data_sets['gold'])
    evaluate_predicted_sentiment(decoded_test_data, gold_data)
    evaluate_summary(decoded_test_data, gold_data, 1)
    evaluate_summary(decoded_test_data, gold_data, 2)


main()
