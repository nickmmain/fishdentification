import cv2
import numpy as np
import matplotlib.pyplot as plt
from fishes import test_image
from show import show_all_frames
from scipy.stats import moment


def histogramFeatures(grayImg):
    # first convert the picture to gray levels, then
    # we need: mean, standard deviation, third moment, fourth moment, contrast, correlation, energy, homogeneity

    # Um what?: ref [6] of Spampinato et al. discusses the usefulness of histograms wrt texture features on pages 850-851
    # The only measures mentioned are: mean, variance, R, standard deviation, third moment, fourth moment, uniformity, average entropy

    # So how are:
    # contrast, correlation, energy, homogeneity (as described in Spampinato et al.)
    # mapped to:
    # variance, R, uniformity, average entropy.

    # Maybe there is a 1-to-1 mapping, maybe not. I will stop at the 4 I know to be accurate, and revisit if I have time.
    histogram = cv2.calcHist([grayImg], [0], None, [256], [0, 256])

    histoFeatures = []
    histoFeatures.append(float(np.mean(histogram)))
    histoFeatures.append(float(np.std(histogram)))
    histoFeatures.append(float(moment(histogram, 3)[0]))
    histoFeatures.append(float(moment(histogram, 4)[0]))

    return ('histoFeatures', histoFeatures)


def contrast(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


def correlation(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


def energy(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries.
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


def homogeneity(img):
    """Some of the values needed to replicate the work of Spampinato et al. are not available as Python libraries. 
     They take some descriptors from section 11.3.3 in Digital Image Processing (3rd Edition) by Gonzalez, Woods"""
    raise NotImplementedError


if __name__ == "__main__":
    img = test_image(True)
    histoFeats = spampinatoHistogramFeatures(img)
    print('hello')
