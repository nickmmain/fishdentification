import cv2
import numpy as np
from fishes import masks

# the images of this data set are very small. The entire contour can be made up of 80 pixels.
numContourPixelsToKeep = 20


def fourier_descriptors_feature(mask):
    # trace the boundary of the bw image:
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours) > 1):
        raise TypeError

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

    spectrum = np.fft.fft(complexValues)

    return contourPixels


if __name__ == "__main__":
    fishMasks = masks(False)
    firstFishMaskPath = fishMasks['mask_01'][76]
    mask = cv2.imread(firstFishMaskPath)
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    withContours = fourier_descriptors_feature(maskCv2Bw)
    cv2.imshow('contours of mask', withContours)
    cv2.waitKey()
