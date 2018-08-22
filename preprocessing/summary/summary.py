#!/Users/liatvi/anaconda3/bin/python
import json
import sys
import subprocess
import os
from textblob import TextBlob
from dataset_utils.dataset_utils import build_data_sets_from_json_file
from preprocessing.summary.pytextrankOpenSource.stage1 import stage1
from preprocessing.summary.pytextrankOpenSource.stage2 import stage2
from preprocessing.summary.pytextrankOpenSource.stage4 import stage4

def extract_corpus_data():
    result = build_data_sets_from_json_file('/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/Corpus/rt-movie-reviews-corpus.json')
    return (result['train'], result['gold'])

def phrases_generator(summarsied_movies_reviews, model_part):
    movie_counter = 0
    movie_phrases = {}
    movie_phrases['movies'] = []

    for movie in summarsied_movies_reviews:
        summary_text = movie['summary']
        summary_text = summary_text.replace('"','')
        summary_text = summary_text.replace('...','')
        movie_phrase = {}
        movie_phrase['name'] = movie['name']
        movie_phrase['summary'] = summary_text

        summary_obj = {}
        summary_obj["id"] = movie_counter
        summary_obj["text"] = summary_text
        movie_phrase['summary_phrases'] = genrate_phrase(summary_obj)

        review_counter = 0
        review_phrases = [] 
        for review in  movie['reviews']:
            review_text = review['text']
            review_text = review_text.replace('"','')
            review_text = review_text.replace('...','')
            review_obj = {}
            review_obj["id"] = review_counter
            review_obj["text"] = review_text
            review_phrases.extend(genrate_phrase(review_obj))
            review_counter += 1

        movie_phrase['reviews_phrases'] = review_phrases
        movie_phrases['movies'].append(movie_phrase)
        # write current filem to the phrases file
        movie_counter += 1

    phrases_outputfile_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/'+ model_part +'phrases.json'
    phrases_file = open(phrases_outputfile_dir , "w+")
    json.dump(movie_phrases, phrases_file)
    phrases_file.close()

def genrate_phrase(text_json):
    json_stage1 = stage1(text_json)
    stage2_result = stage2(json_stage1)
    return stage4(stage2_result)

def main():
    summarsied_movies_reviews_train, summarsied_movies_reviews_gold = extract_corpus_data()
    movie = [x for x in summarsied_movies_reviews_gold if x['id'] == 5909][0]
    del movie['reviews'][81]
    # phrases_generator(summarsied_movies_reviews_train, 'test')
    phrases_generator(summarsied_movies_reviews_gold, 'gold')
    print('vv')
    
main()