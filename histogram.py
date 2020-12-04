import cv2
import numpy as np
import matplotlib.pyplot as plt
from show import show_all_frames
from scipy.stats import moment


def spampinatoHistogramFeatures(img):
    # first convert the picture to gray levels, then
    # we need: mean, standard deviation, third moment, fourth moment, contrast, correlation, energy, homogeneity

    # mean
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([grayImg], [0], None, [256], [0, 256])

    mean = np.mean(histogram)
    stdDeviation = np.std(histogram)
    thirdOrderMoment = moment(histogram, 3)
    fourthOrderMoment = moment(histogram, 4)


def contrast(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""

    raise NotImplementedError


def correlation(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""

    m_r = (mr + i)*p_ij for i in range(len(img.shape[0]))
    m_c = (mc + j)*p_ij for j in range(len(img.shape[1]))

    for i in range(len(img.shape[0])):
        for j in range(len(img.shape[1])):


def energy(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


def homogeneity(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries. 
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


if __name__ == "__main__":
