import json
import cv2
from fishes import fishesAndMasks
from ldaModel import trainModel, testModel

if __name__ == "__main__":
    trainAndTestCount = 3
    results = []

    for i in range(trainAndTestCount):
        # get 300 random pictures of fish, split into training and testing sets
        data = fishesAndMasks(0.7, 300)

        # train the model
        model = trainModel(data)

        # test the model
        result = testModel(model, data)
        results.append(result)

    print('results for training and testing ' +
          str(trainAndTestCount)+' times: '+str(results))
