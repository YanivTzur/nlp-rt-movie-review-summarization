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
    # reviews_json = {}
    # with open('/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/Corpus/rt-movie-reviews-corpus.json', 'r') as corpus_json_file:
    #     reviews_json = json.load(corpus_json_file)
    # summarsied_movies_reviews = [movie for movie in  reviews_json['movies'] if movie['summary'] != 'No consensus yet.']
    # not_summarsied_movies_reviews = [movie for movie in reviews_json['movies'] if movie['summary'] == 'No consensus yet.']
    # return (summarsied_movies_reviews, not_summarsied_movies_reviews)

def delete_files(filename):
    data_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/" + filename
    os.remove(data_dir)
    stage1_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/'+filename
    os.remove(stage1_dir)
    stage2_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/' + filename
    os.remove(stage2_dir)
    phrase_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/phrases/' + filename
    os.remove(phrase_dir)

def phrases_generator(summarsied_movies_reviews, model_part):
    movie_counter = 0
    movie_phrases = {}
    movie_phrases['movies'] = []

    for movie in summarsied_movies_reviews:
        file_name = movie['name'].replace(' ','-')
        summary_file_name = file_name+ "_summary.json"
        # ----------------------------------------------
        # summary_file_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrankOpenSource/dat/" + summary_file_name
        # summary_file = open(summary_file_dir, "w+")
        # ----------------------------------------------
        summary_text = movie['summary']
        summary_text = summary_text.replace('"','')
        # ----------------------------------------------
        # summary_file.write("{\"id\":\"")
        # summary_file.write(str(movie_counter))
        # summary_file.write("\",\"text\":\"")
        # summary_file.write(summary_text)
        # summary_file.write("\"}")
        # summary_file.close()
        # ----------------------------------------------
        movie_phrase = {}
        movie_phrase['name'] = movie['name']
        movie_phrase['summary'] = summary_text

        movie_obj = {}
        movie_obj["id"] = movie_counter
        movie_obj["text"] = summary_text
        movie_phrase['summary_phrases'] = genrate_single_phrase(movie_obj)
        # movie_phrase['summary_phrases'] = genrate_single_phrase(summary_file_dir)
        # delete_files(summary_file_name)

        summary = {}
        summary['id'] = summary_text
        summary['text'] = movie_counter

        review_counter = 0
        review_phrases = [] 
        for review in  movie['reviews']:
            review_file_name = file_name + "_" + str(review_counter) + "_reviews.json"
            reviews_file = open("/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/" + review_file_name , "w+")
            reviews_file.write("{\"id\":\"")
            reviews_file.write(str(review_counter))
            reviews_file.write("\",\"text\":\"")
            review_text = review['text']
            review_text = review_text.replace('"','')
            reviews_file.write(review_text)
            reviews_file.write("\"}")
            reviews_file.close()
            review_phrases.extend(genrate_single_phrase(review_file_name))
            delete_files(review_file_name)
            review_counter += 1

        movie_phrase['review_phrases'] = review_phrases
        movie_phrases['movies'].append(movie_phrase)
        # write current filem to the phrases file
        movie_counter += 1

    phrases_outputfile_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/'+ model_part +'phrases.json'
    phrases_file = open(phrases_outputfile_dir , "w+")
    json.dump(movie_phrases, phrases_file)
    phrases_file.close()

def genrate_single_phrase(text_json):
    # stageOne(filename)
    # stageTwo(filename)
    # # stageTree()
    # stageFour(filename)
    # return get_phrases_array(filename)
    json_stage1 = stage1(text_json)
    stage2_result = stage2(json_stage1)
    rphrases = stage4(stage2_result)
    print('cc')
    return rphrases

def get_phrases_array(filename):
    phrases_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/phrases/'
    with open(phrases_dir + '/' + filename, 'r') as phrases_file:
      return [phrase.strip().replace('\n', '') for phrase in phrases_file.read().split(',')]


def stageOne(filename):
    data_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/dat/'+filename
    output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1/' + filename
    stage1_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage1.py"

    with open(output, "w+") as output:
        subprocess.call(["python3", stage1_dir, data_input], stdout=output)

def stageTwo(filename):
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

def stageFour(filename):
        stage2_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage2/' + filename
        # stage3_input = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage3/' + filename
        output = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/phrases/' + filename
        stage4_dir = "/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/stage4.py"

        with open(output, "w+") as output:
            subprocess.call(["python3", stage4_dir, stage2_input], stdout=output)

# def getParases(summarsied_movies_reviews):
    # phrases_outputfile_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/phrases.json'
    # phrases_dir = '/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/preprocessing/summary/pytextrank/phrases/'
    # film_counter = 0
    # for filename in os.listdir(phrases_dir):
    #     stageOne(filename)
    #     stageTwo(filename)
    #     # stageTree()
    #     stageFour(filename)
    #     with open(phrases_dir + '/' + filename, 'r') as phrases_file:
    #         current_film_phrases = phrases_file.read().split(“,”)



def main():
    summarsied_movies_reviews_train, summarsied_movies_reviews_gold = extract_corpus_data()
    phrases_generator(summarsied_movies_reviews_train, 'test')
    phrases_generator(summarsied_movies_reviews_gold, 'gold')


    # writing to file all the 


main()