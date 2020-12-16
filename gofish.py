import json
import cv2
from fishes import fishesAndMasks
from ldaModel import trainModel, testModel

# testing of the LDA model with 100 pictures of fish

if __name__ == "__main__":
    trainAndTestCount = 10
    results = []

    for i in range(trainAndTestCount):
        # get 300 random pictures of fish, split into training and testing sets
        data = fishesAndMasks(0.7, 300)

        # train the model
        model = trainModel(data)

        result = testModel(model, data)
        results.append(result)

    print('results for training and testing ' +
          str(trainAndTestCount)+' times: '+str(results))
