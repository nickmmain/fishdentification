
import json
import cv2
import pickle
from datetime import datetime
from fishes import fishesAndMasks
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getImageFeatures, getMaskFeatures

# https://machinelearningmastery.com/linear-discriminant-analysis-with-python/


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
            grayMask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            maskFeaturesInArrays = getMaskFeatures(grayMask)

            maskFeaturesSingleArray = [
                item for sublist in maskFeaturesInArrays for item in sublist]

            singleFeaturesArray = imgFeaturesSingleArray+maskFeaturesSingleArray

            features.append(singleFeaturesArray)
            labels.append(imgsFolder)

    # define model
    model = LinearDiscriminantAnalysis()
    # fit model
    return model.fit(features, labels)


def saveModel(model):
    timeNow = datetime.now()
    timeStr = timeNow.strftime('%m%d_%H%M')
    pickle.dump(model, open("ldaModel_"+timeStr+".p", "wb"))


# def loadPreviousModel():


if __name__ == "__main__":
    trainingData = fishesAndMasks(True)

    # TODO: the model should not limit itself to being trained on a subset. limit should be on fishesAndMasks
    model = trainModel(trainingData, 300)

    saveModel(model)
