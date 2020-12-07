
import json
import cv2
import pickle
from datetime import datetime
from fishes import fishes, masks
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getFeaturesArray


def trainModel(fishDict, limit=None):
    features = []
    labels = []

    for imgsFolder in fishDict:
        if not limit:
            limit = len(fishDict[imgsFolder]['train'])
        for imgPath in fishDict[imgsFolder]['train'][0:limit]:
            img = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            featuresInSeveralArrays = getFeaturesArray(grayImg)

            featuresInOneArray = [
                item for sublist in featuresInSeveralArrays for item in sublist]

            features.append(featuresInOneArray)
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
    fishDict = fishes()
    model = trainModel(fishDict, 100)
    saveModel(model)
