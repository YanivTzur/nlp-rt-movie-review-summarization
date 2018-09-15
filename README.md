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
    
    Following is a general outline of how to call the script:
  
  `train.py CORPUS_PATH OUTPUT_FILES_PATH [-mrd {0,1,2}] [-rrd {0,1,2}] [-we {0,1,2}] [-sp {0,1,2}]
                                          [-ts_ss {0,1,2}] [-ts_swe {0,1,2}]`
                                          
  Explanation:
    * Mandatory positional arguments:
      * CORPUS_PATH: The path where the complete corpus is located at on your computer.
      * OUTPUT_FILES_PATH: The directory in which to put the output files (train.json and gold.json).
    * Optional arguments (0 denotes don't compute, 1 denotes use existing if loading from disk, 2 denotes compute from scratch):
      * -mrd: Whether to compute the number of reviews with a certain computed sentiment score, for each of the possible computed values {1,2,3,4,5}.
      * -rrd: Whether to compute the average percent of sentiment phrases in each review with a certain computed sentiment score, for each possible computed value from the set {1,2,3,4,5}.
      * -we: Whether to compute word embeddings for the concatenation of all reviews of each movie.
      * -sp: Whether to compute, for each movie, for each possible sentiment phrase, whether it is present in the reviews of the movie.
      * -ts_ss: Whether to compute the sentiment score for each sentence of each review, for textual summarization.
      * -ts_swe: Whether to compute word embeddings for each sentence of each review, for textual summarization.
  
* decode.py:


* eval.py:

Additionally, there are some more scripts that contain code used by the above-mentioned 3 main scripts. Also there
