import os
import argparse
import re

from utils import collection_utils

parser = argparse.ArgumentParser(description="Conducts a number of experiments equal to the number input by "
                                             "the user. In each experiment, a single trial is conducted for "
                                             "each different combination of parameters.\n"
                                             +
                                             "The program writes the result of each trial, and the mean "
                                             +
                                             "of all trials with the same set of parameters to an output file.")
parser.add_argument("NUM_OF_EXPERIMENTS", help="The number of times to run each trial.",
                    type=int)
parser.add_argument("OUTPUT_FILE_PATH",
                    help="The path of in which to put the output file, including the output file's "
                         "name.", type=str)
args = parser.parse_args()

output_dir_name = os.path.dirname(args.OUTPUT_FILE_PATH)
if not os.path.exists(output_dir_name) or not os.path.isdir(output_dir_name):
    print("Error: no directory exists at the path {}.".format(output_dir_name))

possible_parameters = ['-mrd', '-rrd', '-we', '-sp']
possible_parameters_powerset = list(collection_utils.powerset(possible_parameters))
possible_parameters_powerset.remove(())
results_dictionary = dict()


def get_sentiment_statistics(sentiment_eval_file_lines):
    accuracy_line = sentiment_eval_file_lines[len(sentiment_eval_file_lines) - 2]
    mae_line = sentiment_eval_file_lines[len(sentiment_eval_file_lines) - 1]
    eval_accuracy = float(re.findall('\d\.\d+', accuracy_line)[0])
    eval_mae = float(re.findall('\d\.\d+', mae_line)[0])
    return eval_accuracy, eval_mae


for parameter_combination in possible_parameters_powerset:
    results_dictionary[parameter_combination] = []
    with open(args.OUTPUT_FILE_PATH, 'a') as sentiment_statistics_file:
        sentiment_statistics_file.write('Parameter Combination {{ {} }}:\n'.format(' '.join(parameter_combination)))
    for i in range(0, args.NUM_OF_EXPERIMENTS):
        results_dictionary[parameter_combination].append({'accuracy': -1.0, 'mae': -1.0})
        with open(args.OUTPUT_FILE_PATH, 'a') as sentiment_statistics_file:
            sentiment_statistics_file.write('Experiment No. {}:\n'.format(str(i)))
        os.system(r'python decode.py train.json gold.json C:\implementation ' + ' '.join(parameter_combination))
        os.system(r'python eval.py decoded.json C:\implementation 1 5')
        with open('sentiment_analysis.eval', 'r') as sentiment_eval_file:
            accuracy, mae = get_sentiment_statistics(sentiment_eval_file.readlines())
            results_dictionary[parameter_combination][i]['accuracy'] = accuracy
            results_dictionary[parameter_combination][i]['mae'] = mae
        with open(args.OUTPUT_FILE_PATH, 'a') as sentiment_statistics_file:
            sentiment_statistics_file.write(('Accuracy: {}\n'
                                            +
                                            'MAE: {}\n').format(accuracy, mae))
    with open(args.OUTPUT_FILE_PATH, 'a') as sentiment_statistics_file:
        sentiment_statistics_file.write(('Mean Accuracy: {}\n'
                                        +
                                        'Mean MAE: {}\n\n')
                                        .format(sum(results_dictionary[parameter_combination][i]['accuracy']
                                                    for i in range(0, args.NUM_OF_EXPERIMENTS))
                                                /
                                                args.NUM_OF_EXPERIMENTS,
                                                sum(results_dictionary[parameter_combination][i]['mae']
                                                    for i in range(0, args.NUM_OF_EXPERIMENTS))
                                                /
                                                args.NUM_OF_EXPERIMENTS))
