import cv2
import os
import numpy as np
from fishes import getMasks, dataPath
from contours import checkForMultipleContours


def fourierDescriptorsFeature(mask, contourPointsLimit=None, numFourierDescriptorsToKeep=30):
    '''For the given mask, return the first Fourier Descriptors up to numFourierDescriptorsToKeep.

    Args:
    mask -- a mask, in the cv2.COLOR_RGB2GRAY colorspace
    contourPointsLimit -- For large images, using a limited number of points for the contour is efficient.
    numFourierDescriptorsToKeep -- Number of fourier descriptors to return from the DFT'''

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkForMultipleContours(mask, contours)

    contour = []
    for i in range(len(contours[0])):
        contour.append([item for sublist in contours[0][i]
                        for item in sublist])

    if(contourPointsLimit is None):
        contourPointsLimit = len(contour)

    numContourPixels = len(contour)
    contourPixelIndices = np.linspace(
        0, numContourPixels-1, contourPointsLimit, dtype=np.uint8)

    contourPixels = []
    for i in range(contourPointsLimit):
        contourPixelArray = contour[contourPixelIndices[i]]
        contourPixels.append(contourPixelArray)

    complexValues = []
    for contourPixel in contourPixels:
        complexValues.append(complex(contourPixel[0], contourPixel[1]))

    fullSpectrum = np.fft.fft(complexValues)
    highestOrderFrequencies = fullSpectrum[0:numFourierDescriptorsToKeep]
    absoluteSpectrum = [abs(complexValue)
                        for complexValue in highestOrderFrequencies]

    # translation invariance
    dropFirstTerm = absoluteSpectrum[1:]

    # scale invariance
    dropFirstTermNormalized = [dropFirstTerm[i]/dropFirstTerm[0]
                               for i in range(len(dropFirstTerm))]

    fdFeatures = {}
    fdFeatures["fdFeatures"] = dropFirstTermNormalized

    return fdFeatures


if __name__ == "__main__":
    fishMasks = getMasks(100)
    firstFishMaskPath = fishMasks['mask_01'][75]
    mask = cv2.imread(os.path.join(dataPath, 'mask_01', firstFishMaskPath))
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    withContours = fourierDescriptorsFeature(maskCv2Bw)
