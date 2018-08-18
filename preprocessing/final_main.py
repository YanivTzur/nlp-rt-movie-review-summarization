#!/usr/bin/python
import json
from textblob import TextBlob

def extract_corpus_data():
    reviews_json = {}
    with open('/Users/liatvi/Documents/NLP/final-project-NLP/nlp-rt-movie-review-summarization/Corpus/movies_json_lines.jl', 'r') as corpus_json_file:
        reviews_json = json.load(corpus_json_file)
    return reviews_json['movies']

def toLowerCase(movies):
    for movie in movies:
        movie['name'] = movie['name'].lower()
        movie['summary'] = movie['summary'].lower()
        for review in movie['reviews']:
            review['text'] = review['text'].lower()
    return movies

def getSentimentScore(text):
    blob = TextBlob(text)
    sentenceCount = 0
    textSentimentCount = 0
    for sentence in blob.sentences:
        sentenceCount += 1
        textSentimentCount += sentence.sentiment.polarity
    if textSentimentCount == 0:
        return 0
    textSentimentAvg = textSentimentCount / sentenceCount
    # shift the sentiment score from -1 - 1 to 0 -5 like the overall score
    sentimentScore = round(((textSentimentAvg - (-1)) * (5 - 0)) / (1 - (-1))) + 0
    return sentimentScore

def sentimentAnalysis(movies):
    for movie in movies:
        if movie['summary'] != 'No consensus yet.':
            movie['summary_sentiment_score'] = getSentimentScore(movie['summary'])
        else:
            movie['summary_sentiment_score'] = -1
        
        #reviews score
        for review in movie['reviews']:
            review['text_sentiment_score'] = getSentimentScore(review['text'])

            review['sentences_sentiment_score'] = []
            for sentence in review['sentences']: 
                review['sentences_sentiment_score'].append(getSentimentScore(sentence))
                
            review['words_sentiment_score'] = []
            for word in review['words']:
                review['words_sentiment_score'].append(getSentimentScore(word))
    return movies

def reviewsPreprocessing(movies):
    for movie in movies:
        for review in movie['reviews']:

            #save all sentences as array
            sentences = review['text'].split(".")
            review['sentences'] = [sentence for sentence in sentences if sentence != '']

            #save all the words 
            words = review['text'].split(" ")
            clean_words = []
            for word in words:
                clean_word = word
                for sign in "().{}\\\",":
                    clean_word = clean_word.replace(sign,"")
                clean_words.append(clean_word)
            review['words'] = [word for word in clean_words if word != '']
    return movies

def main():
    movies = extract_corpus_data()
    # movies = movies[1:10]
    movies = toLowerCase(movies)
    movies = reviewsPreprocessing(movies)
    movies = sentimentAnalysis(movies)

main()