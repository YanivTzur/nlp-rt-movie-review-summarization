from pandas.io.json import json_normalize
import json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Importing the dataset and splitting it into training and test sets.

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

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit_generator(X_train,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = X_test,
                         nb_val_samples = 2000)

# End of importing and splitting the dataset