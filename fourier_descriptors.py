import cv2
import numpy as np
from fishes import masks

# the images of this data set are very small. The entire contour can be made up of 80 pixels.
numContourPixelsToKeep = 20
numfourierDescriptorsToKeep = 8


def fourier_descriptors_feature(mask):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    contour = contours[0]
    numContourPixels = len(contour)
    contourPixelIndices = np.linspace(
        0, numContourPixels-1, numContourPixelsToKeep, dtype=np.uint8)
    contourPixels = []

    for i in range(numContourPixelsToKeep):
        contourPixelArray = contour[contourPixelIndices[i]]
        # not sure why this is wrapped in an extra array
        contourPixels.append(contourPixelArray[0])

    complexValues = []
    for contourPixel in contourPixels:
        complexValues.append(complex(contourPixel[0], contourPixel[1]))

    fullSpectrum = np.fft.fft(complexValues)
    firstFrequencies = fullSpectrum[0:numfourierDescriptorsToKeep]
    absSpectrum = [abs(complexValue) for complexValue in firstFrequencies]
    dropFirstTerm = absSpectrum[1:]
    dropFirstTermNormalized = [dropFirstTerm[i]/dropFirstTerm[0]
                               for i in range(len(dropFirstTerm))]

    fdFeatures = {}
    fdFeatures["fdFeatures"] = dropFirstTermNormalized

    return fdFeatures


if __name__ == "__main__":
    fishMasks = masks(False)
    firstFishMaskPath = fishMasks['mask_01'][75]
    mask = cv2.imread(firstFishMaskPath)
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    withContours = fourier_descriptors_feature(maskCv2Bw)
    cv2.imshow('contours of mask', withContours)
    cv2.waitKey()
