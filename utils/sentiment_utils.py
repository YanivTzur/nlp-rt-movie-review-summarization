from textblob import TextBlob
from utils import dataset_utils


def get_k_rating_percentages(text, normalized_range_min, normalized_range_max):
    """
    Receives a certain text, computes the different sentiment phrases (phrases containing
    sentiment bearing words), normalizes the sentiment score of each sentiment phrase so
    it is on a discrete scale from normalized_range_min to normalized_range_max, and returns
    a dictionary containing for each score on this scale the percent of sentiment phrases
    with this score.
    :param text: the text to process as described above.
    :param normalized_range_min: the minimum discrete value of the range into which the
                                 sentiment score of each sentiment phrase is to be converted.
    :param normalized_range_max: the maximum discrete value of the range into which the
                                 sentiment score of each sentiment phrase is to be converted.
    :return: a dictionary as described above.
    """
    rating_percentages = [0] * (normalized_range_max - normalized_range_min + 1)
    text_blob = TextBlob(text)
    assessments = text_blob.sentiment_assessments.assessments
    for sentiment_assessment in assessments:
        normalized_token_sentiment = round(dataset_utils
                                           .shift_scale(sentiment_assessment[1],
                                                        -1, 1,
                                                        normalized_range_min, normalized_range_max))
        rating_percentages[normalized_token_sentiment - 1] += (1 / len(assessments))
    return rating_percentages


def get_sentiment_score(text, normalized_range_min, normalized_range_max):
    """
    Normalizes the sentiment score from a scale of [-1, 1] to the chosen normalized
    discrete scale of {normalized_range_min, normalized_range_min + 1, ..., normalized_range_max}.
    :param text: the input text whose sentiment score is to be computed and returned.
    :param normalized_range_min: the lowest value of the normalized range of scores.
    :param normalized_range_max: the highest value of the normalized range of scores.
    :return: the sentiment score of the input text, normalized to a discrete scale of {1,2,3,4,5}.
    """
    k_rating_percentages = get_k_rating_percentages(text, normalized_range_min, normalized_range_max)
    sentiment_scores = (round(dataset_utils.shift_scale(TextBlob(text).sentiment.polarity, -1, 1,
                                                        normalized_range_min, normalized_range_max)),)
    for rating_percentage in k_rating_percentages:
        sentiment_scores += (rating_percentage,)
    return sentiment_scores
