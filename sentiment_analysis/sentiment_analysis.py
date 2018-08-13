from pandas.io.json import json_normalize
import json
from sklearn.model_selection import train_test_split

# Importing the dataset
input_json = json.load(open('C:\\Users\\yaniv\\Dropbox\\'
                       +
                       'Second Degree - OU\\12. Introduction to Natural Language Processing\\'
                       +
                       'project\\shared\\rt-movie-reviews-corpus.json', 'r'))
dataset = json_normalize(input_json['movies'])
X = dataset.iloc[:, [0, 1, 2, 3, 5, 6]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("Done")