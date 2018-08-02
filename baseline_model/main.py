#!/usr/bin/python
import json
import random
from textblob import TextBlob
from sumeval.metrics.rouge import RougeCalculator

def buildCorpus(jsonDataDir,idAttribute, attributes):
    corpus = {}
    with open(jsonDataDir, 'r') as file_handle:
        for line in file_handle:
            singleReview = json.loads(line)
            singleReviewData = {}
            for attribute in attributes:
                singleReviewData[attribute] = singleReview[attribute]

            if singleReview[idAttribute] not in corpus.keys():
                corpus[singleReview[idAttribute]] = {}
                corpus[singleReview[idAttribute]]['data'] = []
            corpus[singleReview[idAttribute]]['data'].append(singleReviewData)
    return corpus

def buildDataSetsFromCorpus(corpus, testAttributes):
    dataSets = {'train': {}, 'gold': {}, 'test': {}}

    for product in corpus.keys():
        dataSets['train'][product] = {}
        dataSets['gold'][product] = {}
        dataSets['test'][product] = {}
        dataSets['train'][product]['data'] = []
        dataSets['gold'][product]['data'] = []
        dataSets['test'][product]['data'] = []

        # we take from every product 20% of the total amount of reviews for the gold set
        review_count = corpus[product]['data'].__len__()
        dataSets['train'][product]['data'] = corpus[product]['data'][0:round(review_count*0.8)]
        dataSets['gold'][product]['data'] = corpus[product]['data'][round(review_count*0.8) : review_count]
        for specific_review in dataSets['gold'][product]['data']:
            for attribute in testAttributes:
                dataSets['test'][product]['data'].append({attribute: specific_review[attribute]})

    return dataSets

def prepareTrainData(trainDataSet):
    """
    in this section er apply the total score for
    :param trainDataSet: dataSet to be trained
    :return: trained Data set
    """
    for product in trainDataSet:
        sentiScore = 0
        overAllScore = 0
        for retview in trainDataSet[product]['data']:
            overAllScore += retview['overall']

            sentiScore += getSentimentScore(retview['reviewText'])
        overAllScoreAvg = overAllScore / trainDataSet[product]['data'].__len__()
        trainDataSet[product]['overAll'] = overAllScoreAvg

        sentiScoreAvg = sentiScore / trainDataSet[product]['data'].__len__()
        trainDataSet[product]['sentiScore'] = sentiScoreAvg

        # set the ratio between sentiment analysis and aver all score
        trainDataSet[product]['sentiRatio'] = overAllScoreAvg / sentiScoreAvg

        # #set a random sammery as the baseline summery
        # randomReview = random.randrange(0, trainDataSet[product]['data'].__len__())
        # trainDataSet[product]['summary'] = trainDataSet[product]['data'][randomReview]['summary']

    return trainDataSet


def prepareGoldData(goldDataSet):
    for product in goldDataSet:
        sentiScore = 0
        overAllScore = 0
        for retview in goldDataSet[product]['data']:
            overAllScore += retview['overall']
        overAllScoreAvg = round(overAllScore / goldDataSet[product]['data'].__len__())
        goldDataSet[product]['overAll'] = overAllScoreAvg

        # set a the first sammery as the baseline summery
        goldDataSet[product]['summary'] = goldDataSet[product]['data'][0]['summary']

    return goldDataSet

def decode(testData, trainedData):
    for product in testData:
        sentiScore = 0
        for review in testData[product]['data']:
            sentiScore += getSentimentScore(review['reviewText'])
        sentiScoreAvg = sentiScore/ testData[product]['data'].__len__()
        testData[product]['overAll'] = round(sentiScoreAvg * trainedData[product]['sentiRatio'])
        if testData[product]['overAll'] > 5:
            testData[product]['overAll'] = 5

        #set a the first sammery as the baseline summery - we take the first 5 word of a random review as a summer
        if testData[product]['data'][0]['reviewText'].split().__len__() >= 5:
            testData[product]['summary'] = testData[product]['data'][0]['reviewText'].split()[:5]
        else:
            testData[product]['summary'] = testData[product]['data'][0]['reviewText'].split()[:testData[product]['data'][0]['reviewText'].split().__len__()]
        testData[product]['summary'] = ' '.join(testData[product]['summary'])

    return testData

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

def evaluateOverAll(decodedTestData, goldData):
    evalFile = open('baselineOverAllEvaluation', 'w')
    evalFile.writelines(['# ------------------------\n',
                         '#  Over All Evaluation - Final Project - Evaluation\n',
                         '# ------------------------\n',
                         'index\t\t\taccuracy\n'])
    counter = 0
    accuracyPresntagesSum = 0
    for product in decodedTestData:
        accuracyPresntage =((abs(goldData[product]['overAll'] - abs(decodedTestData[product]['overAll'] - goldData[product]['overAll'])))/ goldData[product]['overAll']) * 100
        if accuracyPresntage == 100.0:
            accuracyPresntagesSum += 1
        counter += 1
        evalFile.write(str(counter) + '\t\t\t' + str(accuracyPresntage) + '\n')
    evalFile.write('# ------------------------\n')
    evalFile.write('Over All Average Accuracy:' + str(accuracyPresntagesSum / decodedTestData.__len__()))
    evalFile.close()

def evaluateSummary(decodedTestData, goldData, nGram):
    evalFile = open('baselineESummarevaluation-ROUGE_' + str(nGram), 'w')
    evalFile.writelines(['# ------------------------\n',
                         '#  Summerazion - Rouge_',str(nGram) ,' - Final Project - Evaluation\n',
                         '# ------------------------\n',
                         'index\t\t\tRecall\t\t\tPrecision\t\t\tFscore\n'])
    counter = 0
    RecallCount = 0
    PrecisionCount = 0
    FscoreCount = 0

    for product in decodedTestData:
        rouge = calculateROUGE(goldData[product]["summary"], decodedTestData[product]["summary"], nGram)
        RecallCount += rouge['recall']
        PrecisionCount += rouge['precision']
        FscoreCount += rouge['fScore']

        evalFile.write(str(counter) + '\t\t\t' + str(rouge['recall']) + '\t\t\t' + str(rouge['precision']) + '\t\t\t' + str(rouge['fScore']) + '\n')

    evalFile.write('# ------------------------\n')
    evalFile.write('Average Recall:' + str(RecallCount / decodedTestData.__len__()) + '\n')
    evalFile.write('Average Gold Precision:' + str(PrecisionCount / decodedTestData.__len__())+ '\n')
    evalFile.write('Average Gold FScore:' + str(FscoreCount / decodedTestData.__len__())+ '\n')
    evalFile.close()

def calculateROUGE(summaryGold, summaryTest, nGram):
    rouge = RougeCalculator(stopwords=True, lang="en")

    rouge_recall = rouge.rouge_n(
        summary=summaryTest,
        references=[summaryGold],
        n=nGram, alpha=0)

    rouge_precision = rouge.rouge_n(
        summary=summaryTest,
        references=[summaryGold],
        n=nGram, alpha=1)

    rouge_fScore = rouge.rouge_n(
        summary=summaryTest,
        references=[summaryGold],
        n=nGram, alpha=0.5)
    return {'recall': rouge_recall, 'precision': rouge_precision, 'fScore': rouge_fScore}

def main():
    corpus = buildCorpus("data/reviews_Amazon_Instant_Video_5.json", 'asin', ['reviewText', 'overall', 'summary'])
    dataSets = buildDataSetsFromCorpus(corpus, ['reviewText'])
    trainedData = prepareTrainData(dataSets['train'])
    decodedTestData = decode(dataSets['test'], trainedData)
    goldData = prepareGoldData(dataSets['gold'])
    evaluateOverAll(decodedTestData, goldData)
    evaluateSummary(decodedTestData, goldData, 1)
    evaluateSummary(decodedTestData, goldData, 2)

main()