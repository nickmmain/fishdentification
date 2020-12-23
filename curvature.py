import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from fishes import getMasks, dataPath
from fourier import checkForMultipleContours
from math import ceil

# https://vgg.fiit.stuba.sk/2013-04/css-%E2%80%93-curvature-scale-space-in-opencv/


def curvatureScaleSpaceFeature(mask, numMaxima=10):

    # TODO: getting the contour is common to both fourier + CSS, should be done once for both.
    # Same for checkForMultipleContours()
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkForMultipleContours(contours)
    smoothedContours = smoothing(contours, True)


def smoothing(contours, plot=False):
    X = np.array([contours[0][i][0][0]
                  for i in range(len(contours[0]))]).astype(np.float64)

    Y = np.array([contours[0][i][0][1]
                  for i in range(len(contours[0]))]).astype(np.float64)

    smoothedContours = []
    sigmas = [i+1 for i in range(6)]

    for sigma in sigmas:
        G = cv2.transpose(cv2.getGaussianKernel(10, sigma))
        Xsmooth = cv2.filter2D(X, -1, G)
        Ysmooth = cv2.filter2D(Y, -1, G)

        smoothedContours.append((Xsmooth, Ysmooth))

    if(plot):
        for i in range(len(sigmas)):
            fig = plt.subplot(3, 2, i+1)
            fig.axes.plot(smoothedContours[i][0], smoothedContours[i][1])

        plt.show()

    return smoothedContours


if __name__ == "__main__":
    fishMasks = getMasks(100)
    firstFishMaskPath = fishMasks['mask_01'][74]
    mask = cv2.imread(os.path.join(dataPath, 'mask_01', firstFishMaskPath))
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    curvatureScaleSpaceFeature(maskCv2Bw)
