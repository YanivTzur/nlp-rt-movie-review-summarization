import re

import numpy


def get_float(accuracy_line):
    return float(re.findall('\d\.\d+', accuracy_line)[0])


lines_list = {'1_5': None, '1_3': None}

with open('sentiment_analysis_1_5', 'r') as stats_file:
    lines_list['1_5'] = stats_file.readlines()

with open('sentiment_analysis_1_3', 'r') as stats_file:
    lines_list['1_3'] = stats_file.readlines()

curr_tuple = None

for scale in lines_list:
    dictionary = dict()
    lines = lines_list[scale]
    for i in range(0, len(lines)):
        match_start_of_experiment_set_result = re.findall(r'Parameter Combination { (.+) }:', lines[i])
        if len(match_start_of_experiment_set_result) > 0:
            new_tuple = tuple()
            for component in match_start_of_experiment_set_result[0].split(' '):
                new_tuple += (component,)
            curr_tuple = new_tuple
            if new_tuple not in dictionary.keys():
                dictionary[new_tuple] = []
                continue
        match_start_of_experiment_result = re.findall(r'Experiment No. (\d+)', lines[i])
        if len(match_start_of_experiment_result) > 0:
            dictionary[curr_tuple].append((get_float(lines[i+1]), get_float(lines[i+2])))
    with open('processed_stats_{}.csv'.format(scale), 'w') as output_file:
        output_file.write('use_mrd,use_rrd,use_we,use_sp,experiment_no,accuracy,mae\n')
        curr_string = ''
        for key in dictionary.keys():
            for i in range(0, len(dictionary[key])):
                if '-mrd' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-rrd' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-we' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-sp' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                curr_string += '{},{},{}\n'.format(i, dictionary[key][i][0], dictionary[key][i][1])
                output_file.write(curr_string)
                curr_string = ''
        with open('mean_processed_stats_{}.csv'.format(scale), 'w') as output_file:
            output_file.write('use_mrd,use_rrd,use_we,use_sp,mean_accuracy,mean_mae\n')
            for key in dictionary.keys():
                if '-mrd' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-rrd' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-we' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                if '-sp' in key:
                    curr_string += 'y,'
                else:
                    curr_string += 'n,'
                curr_string += '{},{}\n'.format(float(numpy.mean([dictionary[key][i][0]
                                                      for i in range(0, len(dictionary[key]))])),
                                                float(numpy.mean([dictionary[key][i][1]
                                                      for i in range(0, len(dictionary[key]))])))
                output_file.write(curr_string)
                curr_string = ''
