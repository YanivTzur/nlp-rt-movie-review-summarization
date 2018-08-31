#!/usr/bin/python
import json
from dataset_utils import dataset_utils

def write_to_preprocess_file(movie, index, file_type):
    file_path = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/data/'+ file_type +'/review_' + str(index) + '.txt'
    file_review = open(file_path , "w") 

    summary = movie['summary']
    for review in movie['reviews']:
        file_review.write(review['text'])
        file_review.write('\n')
    file_review.write('\n')
    file_review.write('@highlight\n')
    file_review.write('\n')
    file_review.write(summary)
    
    file_review.close()

def main():
    corpus = dataset_utils.build_data_sets_from_json_file('/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/summary/corpus/rt-movie-reviews-corpus.json')
    index = 0
    for movie in corpus['train']: 
        write_to_preprocess_file(movie, index, 'train')
        index += 1
        
    index = 0
    for movie in corpus['gold']: 
        write_to_preprocess_file(movie, index, 'gold')
        index += 1
    
main()