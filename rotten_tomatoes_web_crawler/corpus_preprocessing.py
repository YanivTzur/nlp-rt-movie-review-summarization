import json
import re
from math import ceil

NORMALIZATION_FACTOR = 5  # Used to fit all ratings to a scale of discrete numbers in the range
                          # [1, NORMALIZATION_FACTOR].
RATING_KEY = 'rating'     # The key in the json object of the numerical rating given to a single review.
SEPARATORS = ['-', '/', '\\'] # Used to separate the actual score from the maximum score of
                                             # a review (except '.', which can also be used but is handled
                                             # separately.

# Used to map each letter grade to a numerical grade.
grade_dictionary = {'Recommended': 5,
                    '*****':5,
                    'A+': 5,
                    'A': 4.5,
                    'A-': 4.5,
                    'B+': 4.25,
                    'B': 4,
                    '****':4,
                    'B-': 3.75,
                    'C+': 3.75,
                    'C': 3.5,
                    'C-': 3.25,
                    'D+': 3.25,
                    'D': 3,
                    '***':3,
                    'D-': 3,
                    'F+': 2,
                    '**':2,
                    'F': 1,
                    'F-': 1,
                    '*': 1,
                    'NotRecommended': 1
                    }


def handle_bad_values(input_movies, input_movie):
    '''
    Removes movies that have no graded reviews and for all other movies, for reviews missing a grade or with a grade
    that's not in the desired range, fills it in with the mean of the grades of the other reviews of the same movies.

    :param input_movies the movies to handle.
    :param input_movie the current inspected movie.
    '''
    input_reviews = input_movie['reviews']
    # Check if all reviews lack a numerical score
    total_non_empty_scores = len([curr_review for curr_review in input_reviews if type(curr_review[RATING_KEY]) is int])
    if total_non_empty_scores == 0 or 'No consensus yet' in input_movie['summary']:
        input_movies.remove(input_movie)
    else:  # At least one review has a score, fill empty values with the mean.
        # Compute mean
        total_score = 0
        for curr_review in input_reviews:
            if type(curr_review[RATING_KEY]) is int \
               and curr_review[RATING_KEY] >= 1 and curr_review[RATING_KEY] <= 5:
                total_score += int(curr_review[RATING_KEY])
        mean = ceil(total_score / total_non_empty_scores)
        # Fill empty/bad values with the mean
        for curr_review in input_reviews:
            if type(curr_review[RATING_KEY]) is str or curr_review[RATING_KEY] < 1 or curr_review[RATING_KEY] > 5:
                curr_review[RATING_KEY] = mean


def split_rating_string(input_rating):
    '''
    Splits grades of a form such as '3/5','3 out of 5', '3 of 5' etc. to its components.
    :param input_rating: the rating string to split.
    :return: the components received from the split.
    '''
    strings_to_search = sorted(SEPARATORS + ['of', 'outof'], key=lambda x: len(x), reverse=True)
    input_rating_components = [input_rating]
    for string_to_search in strings_to_search:
        split_index = "".join(input_rating_components).lower().rfind(string_to_search.lower())
        if split_index > (-1):
            input_rating_components = "".join(input_rating_components).lower().rsplit(string_to_search)
    if '.' in input_rating and input_rating.count('.') > 1:
        # More than one decimal point, as in '1.5.4' denoting a grade
        # of 1.5/4, and opposed to to e.g. '3.0'.
        last_element = input_rating_components[len(input_rating_components)-1]
        input_rating_components += last_element.lower().rsplit('.', 1)
        input_rating_components.remove(last_element)
    return input_rating_components


curr_input_file = open(r'C:\Users\yaniv\Desktop\CorpusGeneratorSpiderProject_backup\rotten_tomatoes_reviews.json',
                       'r')
curr_output_file = open('movies_json_lines_output/preprocessed_rt_reviews.json', 'w')
curr_json = json.load(curr_input_file)
movies = curr_json['movies']
movies_copy = movies[:]

for movie in movies_copy:
    reviews_copy = movie['reviews'][:]
    for review in reviews_copy:
        rating = review[RATING_KEY].strip()
        grade = ''
        if len(rating) > 0:
            rating = re.sub(r"\s*Full Review\s*\|\s*Original Score:\s*", '', rating) # Remove redundant string.
            rating = re.sub(r'\s+|stars', '', rating)  # Remove whitespace and redundant strings.
            rating = re.sub(r'hmoeditup', '', rating)  # Remove more redundant strings.
            rating = re.sub(r'-plus|plus', '+', rating) # Standardize letter grades.
            rating = re.sub(r'-minus|minus', '-', rating) # Standardize letter grades.
            if len(re.findall('\*+\d',rating)) > 0: # Handle strings of the form '***1\2'.
                rating = re.sub(r'\*', '', rating)
            rating = re.sub(r'^-$|^\+$', '', rating)  # Remove empty scores of the form '-'/'+'.
            rating = re.sub(r'-(-)|\+(\+)', '$1', rating)  # Remove repeated '+'/'-'
            rating = re.sub(r"3outta5ain'tgreat", '3/5', rating) # Handle bad string.

            # Map letter grade (e.g. 'A') to numerical grade.
            if len(rating) > 0: # Score still isn't empty.
                lower_case_grade_dictionary = {key.lower():grade_dictionary[key] for key in grade_dictionary.keys()}
                if rating.lower().split('/')[0] in lower_case_grade_dictionary.keys():
                    grade = ceil(lower_case_grade_dictionary[rating.lower().split('/')[0]])
                elif rating.lower().split('\\')[0] in lower_case_grade_dictionary.keys():
                    grade = ceil(lower_case_grade_dictionary[rating.lower().split('\\')[0]])
                # Normalize numerical grade.
                if grade == '':
                    search_result = re.search('\d*(\.\d+)?\D*\d+(\.\d+)?', rating)
                    if search_result is None:
                        grade = ''
                    elif len(rating) > 0:  # Score still isn't empty.
                        rating_components = split_rating_string(rating)
                        try:
                            grade = float(rating_components[0])
                        except:
                            grade = ''
                        if grade != '':
                            if len(rating_components) >= 2:
                                rating_components[1] = re.sub('[^0-9]', '', rating_components[1])
                                if rating_components[1] == '':
                                    max_grade = grade
                                else:
                                    max_grade = float(rating_components[1])
                            else:
                                max_grade = NORMALIZATION_FACTOR
                            if max_grade <= 0: # Handle a max grade of 0, as in '0/0'.
                                grade = 1
                            else:
                                grade = max(1, ceil(grade * (NORMALIZATION_FACTOR / max_grade)))
                                if grade > NORMALIZATION_FACTOR and grade <= 10:
                                    grade = ceil(grade / 10.0 * NORMALIZATION_FACTOR)
                                elif grade > 10 and grade <= 100:
                                    grade = ceil(grade / 100.0 * NORMALIZATION_FACTOR)
        review[RATING_KEY] = grade
    handle_bad_values(movies, movie)
json.dump(curr_json, curr_output_file)
curr_input_file.close()
curr_output_file.close()
