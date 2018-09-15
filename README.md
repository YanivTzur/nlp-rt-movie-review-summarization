# nlp-rt-movie-review-summarization
An NLP project concerned with aggregating movie reviews (such as those on Rotten Tomatoes). The project performs two tasks. One task is sentiment analysis. The second task is the automatic generation of summaries for movies based on their reviews. These tasks are performed based on a dataset of movie reviews crawled from Rotten Tomatoes.

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

  This script takes as input the train.json and gold.json and uses them to predict the sentiment score of each movie, as well as to create a summary for the reviews of each movie. After the script finishes its operation, it creates a json file containing, for each movie, predicted and ground-truth values for both sentiment analysis and textual summarization.
  
  Following is a general outline of how to call the script:
  
  
  `decode.py TRAIN_PATH GOLD_PATH OUTPUT_FILE_PATH NUM_OF_SENTENCES_IN_SUMMARY [-mrd] [-rrd] [-we] [-sp] [-ts_ss] [-ts_swe]
                 [--use-existing]`
  
    Explanation:
    * Mandatory positional arguments:
      * TRAIN_PATH: The path of the train.json file.
      * GOLD_PATH: The path of the gold.json file.
      * OUTPUT_FILE_PATH: The path in which to save the output of the script, including the file's name.
      * NUM_OF_SENTENCES_IN_SUMMARY: The number of sentences to choose from each movie's reviews and construct a summary from.
    * Optional arguments:
      * -mrd: Whether to use the number of reviews with a certain computed sentiment score.
      * -rrd: Whether to use the average percent of sentiment phrases in each review with a certain computed sentiment score.
      * -we: Whether to use word embeddings for the concatenation of all reviews of each movie.
      * -sp: Whether to use, for each movie, for each possible sentiment phrase, whether it is present in the reviews of the movie.
      * -ts_ss: Whether to use the sentiment score for each sentence of each review, for textual summarization.
      * -ts_swe: Whether to use word embeddings for each sentence of each review, for textual summarization.
      * --use-existing: VERY IMPORTANT!!! Use this flag to designate that you want to use a pre-trained neural network model from disk, if such exists for the chosen combination of features. You need to set this flag in order to get the same results as we did in our trained models.

* eval.py:

 This script takes as input the file produced by decode.py, and uses it to evaluate the results of the decoding performed by decode.py. The script produces as output 3 files:
 
   * sentiment_analysis.eval: Shows the results of sentiment analysis.
   * summary_evaluationROUGE1.eval: Shows the ROUGE-1 metrics for the texual summarization task.
   * summary_evaluationROUGE2.eval: Shows the ROUGE-2 metrics for the texual summarization task.

  The script puts these files in the output directory the user gives as input in the command line, as explained below.
  
  Following is a general outline of how to call the script:
  
  
  `eval.py DECODED_PATH OUTPUT_FILES_PATH NORMALIZED_RANGE_MIN NORMALIZED_RANGE_MAX`
  
  Explanation:
  * Mandatory positional arguments:
    * DECODED_PATH: The path of the output of decode.py.
    * OUTPUT_FILES_PATH: The directory where you want the script to put its 3 output files.
    * NORMALIZED_RANGE_MIN: The minimum of the discrete range into which sentiment scores are to be normalized. In our experiments the value of this parameter was always 1.
    * NORMALIZED_RANGE_MAX: The maximum of the discrete range into which sentiment scores are to be normalized. In our experiments the value of this parameter was always 3 or 5.

## Baseline Model Script
This script runs the baseline model on the original complete corpus. To run it you simply need to run:

`baseline_model.py [corpus_path]`

where corpus_path is the path of the corpus on your computer. At the end of its execution, the script produces as output 4 files, similarly to eval.py. These files are:

* baselineOverAllEvaluation_1_3.eval: Contains evaluation of sentiment analysis results on a scale of 1 to 3.
* baselineOverAllEvaluation_1_5.eval: Contains evaluation of sentiment analysis results on a scale of 1 to 5.
* baselineSummaryEvaluationROUGE1.eval: Contains ROUGE-1 metrics of the textual summarization in the baseline model.
* baselineOverAllEvaluation.eval: Contains ROUGE-2 metrics of the textual summarization in the baseline model.

## Web Crawler Code
The folder rotten_tomatoes_web_crawler/CorpusGeneratorSpiderProject contains the code used to crawl Rotten Tomatoes for movie reviews and to use them to construct our corpus. If you are interested, you can find instructions on https://scrapy.org/ regarding how to use their framework to run this web crawling script (also called a spider by scrapy).
Also, the folder rotten_tomatoes_web_crawler contains some ad hoc code created for converting review scores to appropriate values and for testing the results of the crawling.
