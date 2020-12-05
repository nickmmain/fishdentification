from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from features import getFeaturesArray
from fishes import test_image


def trainModel(trainImgs):

    # go get all features for all images
    for img in trainImgs:
        allFeaturesArrays = getFeaturesArray(img)

    # define dataset
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
    # define model
    model = LinearDiscriminantAnalysis()
    # fit model
    model.fit(X, y)


if __name__ == "__main__":
    img = test_image()
    trainModel([img])
