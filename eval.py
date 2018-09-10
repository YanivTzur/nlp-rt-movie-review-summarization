import argparse
import json
import os
import pandas as pd

from utils import dataset_utils


def fill_values(dataframe, column_name, column_values):
    dataframe[column_name] = column_values
    return dataframe


parser = argparse.ArgumentParser(description="Receives a file containing the decoded examples of a "
                                             +
                                             "test set, and produces three evaluation files. The first "
                                             +
                                             "file contains an evaluation of the quality of the "
                                             +
                                             "results of the sentiment analysis of the decoding. The "
                                             +
                                             "second and third files contain an evaluation of the quality of "
                                             +
                                             "the textual summarization of each movie's reviews (using ROUGE-1 and "
                                             +
                                             "and ROUGE-2 metrics).")
parser.add_argument("DECODED_PATH",
                    help="The path of the file containing the decoded examples' data.")
parser.add_argument("OUTPUT_FILES_PATH",
                    help="The path of the directory in which to save the output of "
                         +
                         "the evaluation script.")
parser.add_argument("NORMALIZED_RANGE_MIN",
                    help="The minimum of the discrete range of values into which the sentiment "
                         +
                         " scores are to be converted from a discrete scale of 1-5.",
                    type=int)
parser.add_argument("NORMALIZED_RANGE_MAX",
                    help="The maximum of the discrete range of values into which the sentiment "
                         +
                         " scores are to be converted from a discrete scale of 1-5.",
                    type=int)
args = parser.parse_args()

if not os.path.exists(args.DECODED_PATH):
    print('Error: the file {} does not exist.'.format(args.CORPUS_PATH))
    exit(1)
elif not os.path.exists(args.OUTPUT_FILES_PATH):
    print('Error: the file {} does not exist.'.format(args.OUTPUT_FILES_PATH))
    exit(1)
elif not os.path.isdir(args.OUTPUT_FILES_PATH):
    print('Error: the argument OUTPUT_FILES_PATH must be a directory.'.format(args.OUTPUT_FILES_PATH))
    exit(1)

with open(args.DECODED_PATH, 'r') as test_file:
    decoded_and_gold_data_set = pd.DataFrame(json.load(test_file))

dataset_utils.evaluate_predicted_sentiment(os.path.join(args.OUTPUT_FILES_PATH, "sentiment_analysis.eval"),
                                           decoded_and_gold_data_set
                                           .drop('ground_truth_sentiment', axis=1)
                                           .drop('ground_truth_summary', axis=1)
                                           .to_dict('records'),
                                           decoded_and_gold_data_set
                                           .drop('predicted_sentiment', axis=1)
                                           .drop('predicted_summary', axis=1)
                                           .to_dict('records'),
                                           args.NORMALIZED_RANGE_MIN, args.NORMALIZED_RANGE_MAX,
                                           'predicted_sentiment', 'ground_truth_sentiment')
dataset_utils.evaluate_summary(os.path.join(args.OUTPUT_FILES_PATH, "summary_evaluationROUGE1.eval"),
                               decoded_and_gold_data_set
                               .drop('ground_truth_sentiment', axis=1)
                               .drop('ground_truth_summary', axis=1)
                               .to_dict('records'),
                               decoded_and_gold_data_set
                               .drop('predicted_sentiment', axis=1)
                               .drop('predicted_summary', axis=1)
                               .to_dict('records'),
                               1,
                               'predicted_summary',
                               'ground_truth_summary')
dataset_utils.evaluate_summary(os.path.join(args.OUTPUT_FILES_PATH, "summary_evaluationROUGE2.eval"),
                               decoded_and_gold_data_set
                               .drop('ground_truth_sentiment', axis=1)
                               .drop('ground_truth_summary', axis=1)
                               .to_dict('records'),
                               decoded_and_gold_data_set
                               .drop('predicted_sentiment', axis=1)
                               .drop('predicted_summary', axis=1)
                               .to_dict('records'),
                               2,
                               'predicted_summary',
                               'ground_truth_summary')
