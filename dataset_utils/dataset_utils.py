import json

from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split


def build_corpus(json_data_dir):
    with open(json_data_dir, 'r') as file_handle:
        corpus = json.load(file_handle)
    return corpus


def build_data_sets_from_corpus(corpus):
    data_sets = {'train': [], 'gold': [], 'test': []}

    complete_dataset = json_normalize(corpus['movies'])

    # Splitting the dataset into the Training set and Test set
    training_set, gold_set = train_test_split(complete_dataset,
                                              test_size=0.2, random_state=0)

    data_sets['train'] = training_set.to_dict('records')
    data_sets['gold'] = gold_set.to_dict('records')
    data_sets['test'] = gold_set.drop('summary', axis=1).to_dict('records')

    return data_sets


def build_data_sets_from_json_file(json_file_path):
    corpus = build_corpus(json_file_path)
    return build_data_sets_from_corpus(corpus)
