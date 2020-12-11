import cv2
import numpy as np
from fishes import masks


def fourierDescriptorsFeature(mask, contourPointsLimit=None, numFourierDescriptorsToKeep=30):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkForMultipleContours(contours)

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


def checkForMultipleContours(contours):
    '''Check the contours drawn to make sure there is only 1. 
    If there is more than 1, highlight and display them for debugging.'''

    if(len(contours) > 1):
        # if the masks are properly done, with RETR_EXTERNAL, this code shouldn't be executed.
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        draw_on_these = cv2.cvtColor(np.copy(mask), cv2.COLOR_GRAY2BGR)

        for i in range(len(contours)):
            draw_on_these = cv2.drawContours(
                draw_on_these, contours[i], -1, colors[i % 3], 1)

        window_name = 'more than 1 contour !'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, draw_on_these)
        cv2.waitKey()


if __name__ == "__main__":
    fishMasks = masks(False)
    firstFishMaskPath = fishMasks['mask_01'][75]
    mask = cv2.imread(firstFishMaskPath)
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    withContours = fourierDescriptorsFeature(maskCv2Bw)
    cv2.imshow('contours of mask', withContours)
    cv2.waitKey()
