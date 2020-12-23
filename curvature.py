import cv2
import os
import numpy as np
from fishes import getMasks, dataPath
from fourier import checkForMultipleContours
# https://vgg.fiit.stuba.sk/2013-04/css-%E2%80%93-curvature-scale-space-in-opencv/


def curvatureScaleSpaceFeature(mask, numMaxima=10):

    # TODO: getting the contour is common to both fourier + CSS, should be done once for both.
    # Same for checkForMultipleContours()
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    checkForMultipleContours(contours)
    X = [contours[0][i][0][0] for i in range(len(contours[0]))]
    Y = [contours[0][i][0][1] for i in range(len(contours[0]))]

    #filters = [np.zeros((mask.size), np.uint8) for i in range(5)]

    for sigma in range(5):
        filtur = np.zeros((contours[0].shape[0]), np.uint8)

        cv2.transpose(cv2.getGaussianKernel(
            3, sigma, cv2.CV_64FC1), filtur)

        filtur = np.zeros((mask.size), np.uint8)
        cv2.filter2D(X, Xsmooth, X.depth(), G)
        cv2.filter2D(Y, Ysmooth, Y.depth(), G)


if __name__ == "__main__":
    fishMasks = getMasks(100)
    firstFishMaskPath = fishMasks['mask_01'][75]
    mask = cv2.imread(os.path.join(dataPath, 'mask_01', firstFishMaskPath))
    maskCv2Bw = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    curvatureScaleSpaceFeature(maskCv2Bw)
