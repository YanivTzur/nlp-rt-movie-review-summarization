import json

RATING_KEY = "rating:"
grade_dictionary = {"A+": 10,
                    "A": 9.5,
                    "A-": 9,
                    "B+": 8.5,
                    "B": 8,
                    "B-": 7.5,
                    "C+": 7.5,
                    "C": 7,
                    "C-": 6.5,
                    "D+": 6.5,
                    "D": 6,
                    "D-": 6,
                    "F": 0}

for i in range(1,11):
    curr_input_file = open("movies_json_lines_input/movies_json_lines_{}.jl".format(i), "r")
    curr_output_file = open("movies_json_lines_output/movies_json_lines_{}.jl".format(i), "a")
    lines = curr_input_file.readlines()
    for line in lines:
        curr_json = json.loads(line)
        reviews_copy = curr_json["reviews"][:]
        for review in reviews_copy:
            if len(review[RATING_KEY].strip()) == 0:
                curr_json["reviews"].remove(review)
            else:
                try:
                    rating = review[RATING_KEY].strip()
                    if rating in grade_dictionary.keys():
                        grade = grade_dictionary[rating]
                    else:
                        rating_components = rating.split("/")
                        grade = float(rating_components[0])
                        if len(rating_components) >= 2:
                            max_grade = int(rating_components[1])
                            grade = grade * (10 / max_grade)
                    review[RATING_KEY] = grade
                except:
                    curr_json["reviews"].remove(review)
        json.dump(curr_json, curr_output_file)

