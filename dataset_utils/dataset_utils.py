import json
import os
import datetime
import re
from pandas.io.json import json_normalize
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_PREPROCESSED_PATH = r"C:\Users\yaniv\Dropbox\Second Degree - OU\12. Introduction to Natural Language " \
                         + \
                         r"Processing\project\shared\\"
PREPROCESSED_TRAIN_PATH = BASE_PREPROCESSED_PATH + 'train_preprocessed.json'
PREPROCESSED_TEST_PATH = BASE_PREPROCESSED_PATH + 'test_preprocessed.json'
PREPROCESSED_GOLD_PATH = BASE_PREPROCESSED_PATH + 'gold_preprocessed.json'

nltk.download('stopwords')

english_negation_words = {'no', 'not', 'nothing', 'never', 'nowhere', 'none', 'no one', 'nobody'}
stopwords_to_omit = set(stopwords.words('english')).union(english_negation_words)
ps = PorterStemmer()


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


def preprocess_datasets(data_sets):
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
        with open(BASE_PREPROCESSED_PATH + '{}_preprocessed.json'.format(key), 'w') as output_file:
            json.dump(data_sets[key], output_file)
    print('{}: End of preprocessing'.format(datetime.datetime.now()))
    return data_sets


def build_data_sets_from_corpus(corpus, preprocess=False):
    data_sets = {'train': [], 'gold': [], 'test': []}

    complete_dataset = json_normalize(corpus['movies'])

    # Splitting the dataset into the Training set and Test set
    training_set, gold_set = train_test_split(complete_dataset,
                                              test_size=0.2, random_state=0)

    data_sets['train'] = training_set.to_dict('records')
    data_sets['gold'] = gold_set.to_dict('records')
    data_sets['test'] = gold_set.drop('summary', axis=1)\
                                .drop('average_rating', axis=1)\
                                .to_dict('records')
    if preprocess:
        data_sets = preprocess_datasets(data_sets)
    return data_sets


def preprocessed_data_sets_exist():
    return os.path.exists(PREPROCESSED_TRAIN_PATH) and os.path.exists(PREPROCESSED_TEST_PATH)\
           and os.path.exists(PREPROCESSED_GOLD_PATH)


def get_preprocessed_data_sets():
    return {'train': json.load(open(PREPROCESSED_TRAIN_PATH, 'r')),
            'test': json.load(open(PREPROCESSED_TEST_PATH, 'r')),
            'gold': json.load(open(PREPROCESSED_GOLD_PATH, 'r'))}


def build_data_sets_from_json_file(json_file_path, preprocess=False):
    if preprocess and preprocessed_data_sets_exist():
        return get_preprocessed_data_sets()
    corpus = build_corpus(json_file_path)
    return build_data_sets_from_corpus(corpus, preprocess)
