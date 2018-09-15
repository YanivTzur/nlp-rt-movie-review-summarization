# nlp-rt-movie-review-summarization
An NLP project concerned with aggregating movie reviews (such as those on Rotten Tomatoes). The project receives 

## Installation:
1. Install all needed dependencies. They can all be installed using `pip`. These dependencies include:
    * hashlib
    * BitVector
    * numpy
    * pandas
    * sklearn
    * textblob
    * spacy
    * nltk
    * sumeval
    * tensorflow
    * keras
    * dask
1. Run the command:

   `python -m spacy download en_core_web_lg `

   from a command line window / terminal to download the corpus required for the spacy NLP library to train word embeddings.

## Project Structure:
The project consists of 3 main scripts that you will need to use to make full use of the project and to evaluate its results. These scripts are:

* train.py:
  
  Similarly to what its name suggests, this script is used to generate train and gold sets from the complete corpus (which can be found [here](https://drive.google.com/open?id=1WSc8pYM0f3N_TMHiorClerI91xcm9CvQ).
  The script ultimately creates a training dataset by the name train.json and a gold dataset by the name gold.json. If the need arises, to save computation time, here are links to fully created training and gold datasets:
    * [train.json](https://drive.google.com/open?id=1J4c0YAyrJH2POhpUxk0b3bN8odsgB4W9)
    * [gold.json](https://drive.google.com/open?id=1UbcnIkXsR_aDjxI3CoRV67oecX1FYKnW)
  
* decode.py:


* eval.py:

Additionally, there are some more scripts that contain code used by the above-mentioned 3 main scripts. Also there
