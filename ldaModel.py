
import json
import cv2
from fishes import fishes, masks
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getFeaturesArray


def trainModel(fishDict):
    features = []
    labels = []

    for imgsFolder in fishDict:
        for imgPath in fishDict[imgsFolder]:
            img = cv2.imread(imgPath)
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            features.append(getFeaturesArray(grayImg))
            labels.append(imgsFolder)

    # define model
    model = LinearDiscriminantAnalysis()
    # fit model
    model.fit(features, labels)


if __name__ == "__main__":
    fishDict = fishes()
    trainModel(fishDict)
