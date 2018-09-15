import datetime
import json
import os
import re
from math import floor

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
from sumeval.metrics.rouge import RougeCalculator

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


def drop_redundant_columns(dataset):
    """
    Removes columns from the dataset that are unused in our experiments and returns
    the resulting dataset.
    :param dataset: a dataset.
    :return: the dataset received after removing the redundant columns.
    """
    columns_to_drop = {'year', 'average_rating', 'tomatometer', 'name'} \
        .intersection(list(dataset.columns))
    for column_name in columns_to_drop:
        dataset.drop(column_name, axis=1)
    return dataset


def build_data_sets_from_corpus(corpus):
    """
    Creates training, test and gold sets from the input complete corpus and returns a dictionary
    containing for each name of dataset, the respective dataset.
    :param corpus: the original complete corpus, expected to be received as a json with associated
                   metadata.
    :return: a dictionary of the form {'train':training_set, 'gold':gold_set, 'test':test_set}
    """
    data_sets = {'train': [], 'gold': [], 'test': []}

    complete_dataset = json_normalize(corpus['movies'])

    # Splitting the dataset into the Training set and Test set
    training_set, gold_set = train_test_split(complete_dataset,
                                              test_size=0.2, random_state=0)
    training_set = drop_redundant_columns(training_set)
    gold_set = drop_redundant_columns(gold_set)

    data_sets['train'] = training_set.to_dict('records')
    data_sets['gold'] = gold_set.to_dict('records')
    data_sets['test'] = gold_set.drop('summary', axis=1) \
        .to_dict('records')
    return data_sets


def preprocessed_data_sets_exist(datasets_base_path):
    return os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_TRAIN_FILE_NAME)
                          and os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_TEST_FILE_NAME)) \
                          and os.path.exists(os.path.join(datasets_base_path, PREPROCESSED_GOLD_FILE_NAME)))


def get_preprocessed_data_sets(datasets_directory_name):
    return {'train': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_TRAIN_FILE_NAME), 'r')),
            'test': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_TEST_FILE_NAME), 'r')),
            'gold': json.load(open(os.path.join(datasets_directory_name, PREPROCESSED_GOLD_FILE_NAME), 'r'))}


def get_macro_average_mean_absolute_error(gold_data, movie_rating_lists,
                                          predicted_sentiment_label_name,
                                          ground_truth_label_name):
    """
    Computes the evaluation measure of macro-average mean absolute error, where lower is better.
    :param decoded_test_data: the test set, with its predicted values.
    :param gold_data: the gold set, which has the same examples as the test set, but also has
                      the ground truth values.
    :param movie_rating_lists: a dictionary, containing for each possible sentiment score the list of
                               examples which have that value as their ground truth sentiment score.
    :param predicted_sentiment_label_name: the name of the column where the predicted sentiment is
                                      stored.
    :param ground_truth_label_name: the name of the column where the ground truth sentiment score is
                               stored.
    :return: the macro-average mean absolute error.

    """
    macro_average_mean_absolute_error = 0
    number_of_classes = len(movie_rating_lists.keys())
    normalized_range_min = min(movie_rating_lists.keys())
    normalized_range_max = max(movie_rating_lists.keys()) + 1

    for j in range(normalized_range_min, normalized_range_max):
        if len(movie_rating_lists[j]) > 0:
            for decoded_movie in movie_rating_lists[j]:
                for gold_movie in gold_data:
                    if decoded_movie['id'] == gold_movie['id']:
                        computed_label = round(shift_scale(decoded_movie[predicted_sentiment_label_name], 1,
                                                           5, normalized_range_min, normalized_range_max))
                        ground_truth_label = gold_movie[ground_truth_label_name]
                        macro_average_mean_absolute_error += abs(computed_label - ground_truth_label)
            macro_average_mean_absolute_error *= (1.0 / len(movie_rating_lists[j]))
    macro_average_mean_absolute_error *= (1.0 * number_of_classes)
    return macro_average_mean_absolute_error


def evaluate_predicted_sentiment(evaluation_file_name,
                                 decoded_test_data, gold_data,
                                 normalized_range_min, normalized_range_max,
                                 predicted_sentiment_label_name='average_rating',
                                 ground_truth_label_name='rating_label'):
    """
    Evaluates the accuracy of the predicted sentiment for each movie in the test set by comparing
    to the gold set.
    The method also accepts as input two integers normalized_range_min and normalized_range_max,
    such that normalized_range_max > normalized_range_min, and first converts each test and gold
    rating to the respective normalized scale before comparing them.

    :param evaluation_file_name: the name to give the output file.
    :param decoded_test_data: the test set data.
    :param gold_data: the gold set data.
    :param normalized_range_min: the minimum value in the range to normalize ratings to.
    :param normalized_range_max: the maximum value in the range to normalize ratings to.
    :param predicted_sentiment_label_name: the name of the column where the predicted sentiment is
                                      stored.
    :param ground_truth_label_name: the name of the column where the ground truth sentiment score is
                               stored.
    :return: the evaluated accuracy of the prediction as a percentage.
    """
    eval_file = open(evaluation_file_name, 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Over All Evaluation - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\tmovie id\tpredicted score\treal score\taccuracy\n'])
    counter = 0
    accuracy_percentage_sum = 0
    curr_num_of_correct_predictions = 0
    movie_rating_lists = {key: [] for key in range(normalized_range_min, normalized_range_max + 1)}

    computed_label = -1
    ground_truth_label = -1

    for decoded_movie in decoded_test_data:
        for gold_movie in gold_data:
            if decoded_movie['id'] == gold_movie['id']:
                if 'reviews' in gold_movie.keys():
                    review_ratings = [review['rating']
                                      for review in gold_movie['reviews']]
                    gold_movie[ground_truth_label_name] = floor(shift_scale(round(sum(review_ratings)
                                                                                  /
                                                                                  len(review_ratings)),
                                                                            1, 5, normalized_range_min,
                                                                            normalized_range_max))
                else:
                    gold_movie[ground_truth_label_name] = floor(shift_scale(gold_movie[ground_truth_label_name],
                                                                            1, 5, normalized_range_min,
                                                                            normalized_range_max))
                computed_label = floor(shift_scale(decoded_movie[predicted_sentiment_label_name], 1,
                                                   5, normalized_range_min, normalized_range_max))
                ground_truth_label = gold_movie[ground_truth_label_name]
                movie_rating_lists[ground_truth_label].append(decoded_movie)
                if computed_label != ground_truth_label:
                    curr_num_of_correct_predictions = 0
                else:
                    curr_num_of_correct_predictions = 1
                    accuracy_percentage_sum += 1
        counter += 1
        eval_file.write("{:<5d}\t{:<8d}\t{:<15d}\t{:<10d}\t{:<8d}\n".format(counter,
                                                                            decoded_movie['id'],
                                                                            computed_label,
                                                                            ground_truth_label,
                                                                            curr_num_of_correct_predictions))
        # eval_file.write("{}\n".format("\t\t\t\t".join([str(counter),
        #                                                str(curr_num_of_correct_predictions)])))
    accuracy_percentage = (accuracy_percentage_sum * 1.0) / counter
    eval_file.write('# ------------------------\n')
    eval_file.write('Overall Accuracy (higher is better): {}\n'.format(str(accuracy_percentage)))
    eval_file.write('Macro-Average Mean Absolute Error (lower is better): '
                    +
                    str(get_macro_average_mean_absolute_error(gold_data,
                                                              movie_rating_lists,
                                                              predicted_sentiment_label_name,
                                                              ground_truth_label_name)))
    eval_file.close()

    return accuracy_percentage


def evaluate_summary(eval_file_name, decoded_test_data, gold_data, n_gram_order,
                     predicted_summary_label='summary',
                     ground_truth_summary_label='summary'):
    eval_file = open(eval_file_name, 'w')
    eval_file.writelines(['# ------------------------\n',
                          '#  Summarization - Rouge_', str(n_gram_order), ' - Final Project - Evaluation\n',
                          '# ------------------------\n',
                          'index\t\t\tRecall\t\t\tPrecision\t\t\tFscore\n'])
    counter = 1
    recall_sum = 0
    precision_sum = 0
    f_score_sum = 0
    test_to_gold_map = dict()
    for decoded_movie in decoded_test_data:
        for gold_movie in gold_data:
            if decoded_movie['id'] == gold_movie['id']:
                test_to_gold_map[decoded_movie['id']] = (decoded_movie, gold_movie)

    for decoded_movie_id in test_to_gold_map.keys():
        rouge = calculate_rouge(preprocess_text(test_to_gold_map[decoded_movie_id][1][ground_truth_summary_label]),
                                preprocess_text(test_to_gold_map[decoded_movie_id][0][predicted_summary_label]),
                                n_gram_order)
        curr_recall = rouge['recall']
        curr_precision = rouge['precision']
        curr_f_score = rouge['fScore']
        recall_sum += curr_recall
        precision_sum += curr_precision
        f_score_sum += curr_f_score

        eval_file.write("{}\n".format("\t\t\t".join([str(counter),
                                                     str(curr_recall),
                                                     str(curr_precision),
                                                     str(curr_f_score)])))
        counter += 1

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


def build_data_sets_from_json_file(json_file_path):
    # datasets_directory_name = os.path.dirname(json_file_path)
    corpus = build_corpus(json_file_path)
    return build_data_sets_from_corpus(corpus)
