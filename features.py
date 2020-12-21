import cv2
import os
from glcm import mahotas_glcmFeatures
from gabor import gaborFeatures
from histogram import histogramFeatures
from fourier import fourierDescriptorsFeature


def getFeatures(data, trainOrTest='train'):
    '''For the given data this function returns all features computable by this project. 

    Args: 
    data -- dictionary of form described in datadict.json
    trainOrtest -- to use the "train" or "test" key of the data dict. Not too proud of this.'''

    features = []
    labels = []

    fishData = data['fish']
    fishDataKeys = list(fishData.keys())
    masksData = data['masks']
    masksDataKeys = list(masksData.keys())

    for i in range(len(fishData)):
        fishTypeTrainingMasks = masksData[masksDataKeys[i]][trainOrTest]
        fishTypeTrainingImgs = fishData[fishDataKeys[i]][trainOrTest]

        for j in range(len(fishTypeTrainingImgs)):
            maskPath = os.path.join(
                data['data_dir'], masksDataKeys[i], fishTypeTrainingMasks[j])
            fishPath = os.path.join(
                data['data_dir'], fishDataKeys[i], fishTypeTrainingImgs[j])

            # read image and mask
            mask = cv2.imread(maskPath)
            img = cv2.imread(fishPath)

            # apply mask to image
            grayMask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            img = cv2.bitwise_and(img, img, mask=grayMask)

            # get image features
            grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgFeaturesInArrays = getImageFeatures(grayImg)
            imgFeaturesSingleArray = [
                item for sublist in imgFeaturesInArrays for item in sublist]

            # get mask features
            maskFeaturesInArrays = getMaskFeatures(grayMask)
            maskFeaturesSingleArray = [
                item for sublist in maskFeaturesInArrays for item in sublist]

            # add all features for photo, and label for this image
            singleFeaturesArray = imgFeaturesSingleArray+maskFeaturesSingleArray
            features.append(singleFeaturesArray)
            labels.append(fishDataKeys[i])

    return features, labels


def getImageFeatures(img):
    '''For the given image this function returns the array of all features computable by this project'''

    features = []

    gf = gaborFeatures(img)
    features.append(gf['gaborFeatures']['mean'])
    features.append(gf['gaborFeatures']['stdDeviation'])

    histo = histogramFeatures(img)
    features.append(histo[1])

    glcm = mahotas_glcmFeatures(img)
    features.append(glcm[1])

    return features


def getMaskFeatures(mask):
    '''For the given mask this function returns the array of all features computable by this project'''

    features = []

    # this works:
    # fourierDscptrs = fourierDescriptorsFeature(mask, 20, 8)

    # this doesn't:
    # fourierDscptrs = fourierDescriptorsFeature(mask, None, 30)

    # this does work:
    # fourierDscptrs = fourierDescriptorsFeature(mask, None, 8)

    # I think this indicates that some images don't even have 30 contour points.
    # so let's try to retain 20 instead:
    fourierDscptrs = fourierDescriptorsFeature(mask, None, 10)
    # ^ it works.

    features.append(fourierDscptrs["fdFeatures"])

    return features


def getFeaturesDictionary():
    '''For the given image this function returns a dictionary of all features computable by this project'''

    features = {}
    features['texture'] = {}

    gf = gaborFeatures(img)
    features['texture'].update(gf)

    glcm = mahotas_glcmFeatures(img)
    features['texture'][glcm[0]] = glcm[1]

    histo = spampinatoHistogramFeatures(img)
    features['texture'][histo[0]] = histo[1]

    return features
