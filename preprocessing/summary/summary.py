#!/usr/bin/python
import json
import sys
import subprocess
import os
from textblob import TextBlob

def extract_corpus_data():
    reviews_json = {}
    with open('/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/Corpus/rt-movie-reviews-corpus.json', 'r') as corpus_json_file:
        reviews_json = json.load(corpus_json_file)
    summarsied_movies_reviews = [movie for movie in  reviews_json['movies'] if movie['summary'] != 'No consensus yet.']
    not_summarsied_movies_reviews = [movie for movie in reviews_json['movies'] if movie['summary'] == 'No consensus yet.']
    return (summarsied_movies_reviews, not_summarsied_movies_reviews)

def writeReviewsToTextFiles(summarsied_movies_reviews):
    movie_counter = 0
    for movie in summarsied_movies_reviews:
        file_name = movie['name'].replace(' ','-')
        summary_file = open("/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/" + file_name + "_summary.json" , "w+")
        summary_text = movie['summary']
        summary_text = summary_text.replace('"','')
        summary_file.write("{\"id\":\"")
        summary_file.write(str(movie_counter))
        summary_file.write("\",\"text\":\"")
        summary_file.write(summary_text)
        summary_file.write("\"}")

        review_counter = 0
        for review in  movie['reviews']:
            reviews_file = open("/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/" + file_name + "_" + str(review_counter) + "_reviews.json" , "w+")
            reviews_file.write("{\"id\":\"")
            reviews_file.write(str(review_counter))
            reviews_file.write("\",\"text\":\"")
            review_text = review['text']
            review_text = review_text.replace('"','')
            reviews_file.write(review_text)
            reviews_file.write("\"}")
            review_counter += 1
        movie_counter += 1

def stageOne(filename):
    file_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/'
    for filename in os.listdir(file_dir):

        data_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/'+filename
        output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/' + filename
        stage1_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1.py"

        with open(output, "w+") as output:
            subprocess.call(["python3", stage1_dir, data_input], stdout=output)

def stageTwo():
    file_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/'
    for filename in os.listdir(file_dir):

        stage1_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/'+filename
        output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/' + filename
        stage2_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2.py"

        with open(output, "w+") as output:
            subprocess.call(["python3", stage2_dir, stage1_input], stdout=output)

#--- stage for the summery constraction - not nessery fro phrases extraction
# def stageTree():
#     file_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/'
#     for filename in os.listdir(file_dir):

#         stage1_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/'+filename
#         stage2_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/'+filename
#         output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage3/' + filename
#         stage3_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage3.py"

#         with open(output, "w+") as output:
#             subprocess.call(["python3", stage3_dir, stage1_input, stage2_input], stdout=output)

def stageFour():
    file_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/'
    for filename in os.listdir(file_dir):

        stage2_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/' + filename
        # stage3_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage3/' + filename
        output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/phrases/' + filename
        stage4_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage4.py"

        with open(output, "w+") as output:
            subprocess.call(["python3", stage4_dir, stage2_input], stdout=output)

def getParases():
    file_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/'
    for filename in os.listdir(file_dir):
        stageOne()
        stageTwo()
        # stageTree()
        stageFour()

###########
# output the movie phrases to a file
###########
def output_phrases():
    output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/phrases.json




def main():
    (summarsied_movies_reviews, not_summarsied_movies_reviews) = extract_corpus_data()
    summarsied_movies_reviews = summarsied_movies_reviews[1:10]
    writeReviewsToTextFiles(summarsied_movies_reviews)
    getParases()

    # writing to file all the 


main()