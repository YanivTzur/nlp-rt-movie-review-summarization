import json
import os
import datetime
import re
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sumeval.metrics.rouge import RougeCalculator
import matplotlib.pyplot as plt

PREPROCESSED_TRAIN_FILE_NAME = 'train_preprocessed.json'
PREPROCESSED_TEST_FILE_NAME = 'test_preprocessed.json'
PREPROCESSED_GOLD_FILE_NAME = 'gold_preprocessed.json'

nltk.download('stopwords')

english_negation_words = {'no', 'not', 'nothing', 'never', 'nowhere', 'none', 'no one', 'nobody'}
stopwords_to_omit = set(stopwords.words('english')).union(english_negation_words)
ps = PorterStemmer()


def shift_scale(old_value, old_min, old_max, new_min, new_max):
    return ((new_max - new_min) / (old_max - old_min)) * (old_value - old_max) + new_max;


def build_corpus(json_data_dir):
    with open(json_data_dir, 'r') as file_handle:
        corpus = json.load(file_handle)
    return corpus


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


def preprocess_datasets(data_sets, datasets_base_path):
    print('{}: Starting to preprocess datasets'.format(datetime.datetime.now()))
    movie_count = 0
    for key in data_sets.keys():
        for movie in data_sets[key]:
            if 'summary' in movie.keys():
                movie['summary'] = preprocess_text(movie['summary'])
            for review in movie['reviews']:
                review['text'] = preprocess_text(review['text'])
            movie_count += 1
            print("Movies preprocessed: {}".format(movie_count))
        with open(os.path.join(datasets_base_path, '{}_preprocessed.json'.format(key)), 'w') as output_file:
            json.dump(data_sets[key], output_file)
    print('{}: End of preprocessing'.format(datetime.datetime.now()))
    return data_sets


def build_data_sets_from_corpus(datasets_directory_name, corpus, preprocess=False):
    data_sets = {'train': [], 'gold': [], 'test': []}

    complete_dataset = json_normalize(corpus['movies'])

    # Splitting the dataset into the Training set and Test set
    training_set, gold_set = train_test_split(complete_dataset,
                                              test_size=0.2, random_state=0)

    data_sets['train'] = training_set.to_dict('records')
    data_sets['gold'] = gold_set.to_dict('records')
    data_sets['test'] = gold_set.drop('summary', axis=1) \
        .drop('average_rating', axis=1) \
        .to_dict('records')
    if preprocess:
        data_sets = preprocess_datasets(data_sets, datasets_directory_name)
    return data_sets


def preprocessed_data_sets_exist(datasets_base_path):
    return os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_TRAIN_FILE_NAME)
                          and os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_TEST_FILE_NAME)) \
                          and os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_GOLD_FILE_NAME)))


def get_preprocessed_data_sets(datasets_directory_name):
    return {'train': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_TRAIN_FILE_NAME), 'r')),
            'test': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_TEST_FILE_NAME), 'r')),
            'gold': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_GOLD_FILE_NAME), 'r'))}


def evaluate_predicted_sentiment(decoded_test_data, gold_data, normalized_range_min,
                                 normalized_range_max):
    """
    Evaluates the accuracy of the predicted sentiment for each movie in the test set by comparing
    to the gold set.
    The method also accepts as input two integers normalized_range_min and normalized_range_max,
    such that normalized_range_max > normalized_range_min, and first converts each test and gold
    rating to the respective normalized scale before comparing them.
    :param decoded_test_data: the test set data.
    :param gold_data: the gold set data.
    :param normalized_range_min: the minimum value in the range to normalize ratings to.
    :param normalized_range_max: the maximum value in the range to normalize ratings to.
    :return: the evaluated accuracy of the prediction as a percentage.
    """
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
                review_ratings = [review['rating']
                                  for review in gold_movie['reviews']]
                gold_movie['rating_label'] = round(shift_scale(round(sum(review_ratings)
                                                                     /
                                                                     len(review_ratings)),
                                                               1, 5, normalized_range_min,
                                                               normalized_range_max))
                computed_label = round(shift_scale(decoded_movie['average_rating'], 1,
                                       5, normalized_range_min, normalized_range_max))
                ground_truth_label = gold_movie['rating_label']
                if computed_label == ground_truth_label:
                    accuracy_percentage_sum += 1
        counter += 1
        accuracy_percentage = (accuracy_percentage_sum * 1.0) / counter
        eval_file.write(str(counter) + '\t\t\t' + str(accuracy_percentage) + '\n')
    eval_file.write('# ------------------------\n')
    eval_file.write('Overall Average Accuracy:' + str(accuracy_percentage))
    eval_file.close()

    return accuracy_percentage


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
        rouge = calculate_rouge(preprocess_text(test_to_gold_map[decoded_movie_id][1]["summary"]),
                                preprocess_text(test_to_gold_map[decoded_movie_id][0]["summary"]),
                                n_gram_order)
        recall_sum += rouge['recall']
        precision_sum += rouge['precision']
        f_score_sum += rouge['fScore']

        eval_file.write(
            str(counter) + '\t\t\t' + str(rouge['recall']) + '\t\t\t' + str(rouge['precision']) + '\t\t\t' + str(
                rouge['fScore']) + '\n')

    recall = recall_sum / len(decoded_test_data)
    precision = precision_sum / len(decoded_test_data)
    fscore = f_score_sum / len(decoded_test_data)
    eval_file.write('# ------------------------\n')
    eval_file.write('Average Recall:' + str(recall) + '\n')
    eval_file.write('Average Gold Precision:' + str(precision) + '\n')
    eval_file.write('Average Gold FScore:' + str(fscore) + '\n')
    eval_file.close()
    return recall, precision, fscore


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


def build_data_sets_from_json_file(json_file_path, preprocess=False):
    datasets_directory_name = os.path.dirname(json_file_path)
    if preprocess and preprocessed_data_sets_exist(datasets_directory_name):
        return get_preprocessed_data_sets(datasets_directory_name)
    corpus = build_corpus(json_file_path)
    return build_data_sets_from_corpus(datasets_directory_name, corpus, preprocess)
