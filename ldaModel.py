
import json
import cv2
import pickle
import os
from datetime import datetime, timezone
from fishes import fishesAndMasks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getFeatures

# https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
savedModelsTimeFormat = '%Y_%m%d_%H%M'
modelDirectory = os.path.join(os.getcwd(), 'models')


def trainModel(trainingData, limit=None):
    features, labels = getFeatures(trainingData, 'train', limit)
    model = LinearDiscriminantAnalysis()
    return model.fit(features, labels)


def testModel(model, testData, limit=None):
    features, labels = getFeatures(testData, 'test', limit)
    predictions = model.predict(features)

    correct = 0
    incorrect = 0
    for i in range(len(labels)):
        if(labels[i] == predictions[i]):
            correct += 1
        else:
            incorrect += 1

    return correct/(correct+incorrect)


def saveModel(model):
    timeNow = datetime.now(timezone.utc)
    timeStr = timeNow.strftime(savedModelsTimeFormat)
    pickle.dump(model, open(os.path.join(
        modelDirectory, "ldaModel_"+timeStr+".p"), "wb"))


def loadPreviousModel():
    newest = datetime(1970, 1, 1)

    for (dirpath, dirnames, filenames) in os.walk(modelDirectory):
        for filname in filenames:
            timestamp = filname.split('_', 1)[1].split('.')[0]
            modelDate = datetime.strptime(timestamp, savedModelsTimeFormat)
            if modelDate > newest:
                newest = modelDate

    timeStr = newest.strftime(savedModelsTimeFormat)
    mostRecentModelPath = os.path.join(
        modelDirectory, "ldaModel_"+timeStr+".p")

    return pickle.loads(open(mostRecentModelPath, 'rb').read())


if __name__ == "__main__":
    trainingData = fishesAndMasks(True)
    # TODO: the model should not limit itself to being trained on a subset. limit should be on fishesAndMasks
    model = trainModel(trainingData, 300)
    saveModel(model)

    # model = loadPreviousModel()
    # testingData = fishesAndMasks(True)
    # rate = testModel(model, testingData, 100)
    # print("Correct predictions rate:"+str(rate))
