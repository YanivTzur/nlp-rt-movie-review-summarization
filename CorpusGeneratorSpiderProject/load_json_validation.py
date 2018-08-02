import json

json_corpus = None
with open('shared/movies_json_lines.jl','r') as input_file:
    json_corpus = json.load(input_file)
movies = json_corpus['movies']
print('Number of movies: {0}'.format(len(movies)))
movies_without_reviews = [movie for movie in movies if len(movie['reviews']) == 0]
print('Number of movies without reviews: {0}'.format(len(movies_without_reviews)))
for movie_without_review in movies_without_reviews:
    for movie in movies:
        if movie_without_review['id'] == movie['id']:
            movies.remove(movie)
with open('shared/movies_json_lines.jl', 'w') as output_file:
    json.dump(json_corpus, output_file)
