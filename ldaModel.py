
import json
import cv2
import pickle
import os
from datetime import datetime, timezone
from fishes import fishesAndMasks
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getImageFeatures, getMaskFeatures

# https://machinelearningmastery.com/linear-discriminant-analysis-with-python/
savedModelsTimeFormat = '%Y_%m%d_%H%M'
modelDirectory = os.path.join(os.getcwd(), 'models')


def trainModel(trainingData, limit=None):
    features = []
    labels = []

    fishData = trainingData['fish']
    fishDataKeys = list(fishData.keys())
    masksData = trainingData['masks']
    masksDataKeys = list(masksData.keys())

    for i in range(len(fishData)):
        fishTypeTrainingImgs = fishData[fishDataKeys[i]]['train']
        fishTypeTrainingMasks = masksData[masksDataKeys[i]]['train']
        if not limit:
            limit = len(fishTypeTrainingImgs)
        for j in range(len(fishTypeTrainingImgs[0:limit])):

            img = cv2.imread(fishTypeTrainingImgs[j])
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgFeaturesInArrays = getImageFeatures(grayImg)

            imgFeaturesSingleArray = [
                item for sublist in imgFeaturesInArrays for item in sublist]

            mask = cv2.imread(fishTypeTrainingMasks[j])
            grayMask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            maskFeaturesInArrays = getMaskFeatures(grayMask)

            maskFeaturesSingleArray = [
                item for sublist in maskFeaturesInArrays for item in sublist]

            singleFeaturesArray = imgFeaturesSingleArray+maskFeaturesSingleArray

            features.append(singleFeaturesArray)
            labels.append(fishDataKeys[i])

    # define model
    model = LinearDiscriminantAnalysis()
    # fit model
    return model.fit(features, labels)


def testModel(model, testData, limit=None):
    features = []
    labels = []

    fishData = testData['fish']
    fishDataKeys = list(fishData.keys())
    masksData = testData['masks']
    masksDataKeys = list(masksData.keys())

    for i in range(len(fishData)):
        fishTypeTrainingImgs = fishData[fishDataKeys[i]]['test']
        fishTypeTrainingMasks = masksData[masksDataKeys[i]]['test']
        if not limit:
            limit = len(fishTypeTrainingImgs)
        for j in range(len(fishTypeTrainingImgs[0:limit])):

            img = cv2.imread(fishTypeTrainingImgs[j])
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgFeaturesInArrays = getImageFeatures(grayImg)

            imgFeaturesSingleArray = [
                item for sublist in imgFeaturesInArrays for item in sublist]

            mask = cv2.imread(fishTypeTrainingMasks[j])
            grayMask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            maskFeaturesInArrays = getMaskFeatures(grayMask)

            maskFeaturesSingleArray = [
                item for sublist in maskFeaturesInArrays for item in sublist]

            singleFeaturesArray = imgFeaturesSingleArray+maskFeaturesSingleArray

            features.append(singleFeaturesArray)
            labels.append(fishDataKeys[i])

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
    # trainingData = fishesAndMasks(True)
    # TODO: the model should not limit itself to being trained on a subset. limit should be on fishesAndMasks
    # model = trainModel(trainingData, 300)
    # saveModel(model)

    model = loadPreviousModel()
    testingData = fishesAndMasks(True)
    rate = testModel(model, testingData, 100)
    print("Correct predictions rate:"+str(rate))
