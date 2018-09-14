import os
import json
from datetime import datetime

train_movie_list = []

files = os.listdir('./output')
for file in files:
    print("{}: Start of processing input file {}.".format(datetime.now(), file))
    with open(os.path.join('output',file), 'r') as input_file:
        movie_list = json.load(input_file)
        movie_list = movie_list[0]
        counter = 0
        for i in range(0, len(movie_list)):
            movie_list[i]['sentences_data'] = movie_list[i]['sentences_data'][0]
            for sentence_data in movie_list[i]['sentences_data']:
                for j in range(0, len(sentence_data[2])):
                    sentence_data[2][j] = round(sentence_data[2][j], 3)
            if 'train' in file:
                train_movie_list.append(movie_list[i])
            print("{}: Processed movie No. {}.".format(datetime.now(), str(counter)))
            counter += 1
    print("{}: End of processing input file {}.".format(datetime.now(), file))
    if 'gold' in file:
        print("{}: Start of dumping gold json file.".format(datetime.now()))
        with open('gold_output.json', 'w') as output_gold_file:
            json.dump(movie_list, output_gold_file)
        print("{}: End of dumping gold json file.".format(datetime.now()))

with open('train_output.json', 'w') as output_train_file:
    print("{}: Start of dumping train json file.".format(datetime.now()))
    json.dump(train_movie_list, output_train_file)
    print("{}: Start of dumping train json file.".format(datetime.now()))
