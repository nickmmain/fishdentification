from glcm import mahotas_glcmFeatures
from gabor import gaborFeatures
from histogram import histogramFeatures
from fourier import fourierDescriptorsFeature


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
    features = []

    fourierDscptrs = fourierDescriptorsFeature(mask)
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
