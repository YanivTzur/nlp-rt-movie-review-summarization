import json

json_corpus = None
with open('movies_json_lines_output/preprocessed_rt_reviews.json', 'r') as input_file:
    json_corpus = json.load(input_file)
movies = json_corpus['movies']
print('Lowest year: {}'.format(min([movie['year'] for movie in movies])))
print('Highest year: {}'.format(max([movie['year'] for movie in movies])))
print('Number of movies: {0}'.format(len(movies)))
movies_without_reviews = [movie for movie in movies if len(movie['reviews']) == 0]
print('Number of movies without reviews: {0}'.format(len(movies_without_reviews)))
print('Total number of reviews: {0}'.format(len([review for movie in movies for review in movie['reviews']])))
reviews_with_bad_score = [review for movie in movies for review in movie['reviews']
                          if type(review['rating']) is not int or review['rating'] not in range(1,6)]
print('Total number of reviews with bad score: {0}'.format(len(reviews_with_bad_score)))
for i in range (1,6):
    print('Total number of reviews with score {0}: {1}'.format(i, len([review for movie in movies
                                                                       for review in movie['reviews']
                                                                       if review['rating'] == i])))
movie_scores = {1:0, 2:0, 3:0, 4:0, 5:0}
for movie in movies:
    movie_ratings = [review['rating'] for review in movie['reviews']]
    computed_average_rating = round(sum(movie_ratings) / len(movie_ratings))
    movie_scores[computed_average_rating] = movie_scores[computed_average_rating] + 1
for i in range (1,6):
    print('Total number of movies with average score {0}: {1}'.format(i, movie_scores[i]))
# for movie_without_review in movies_without_reviews:
#     for movie in movies:
#         if movie_without_review['id'] == movie['id']:
#             movies.remove(movie)
# with open('shared/movies_json_lines.jl', 'w') as output_file:
#     json.dump(json_corpus, output_file)