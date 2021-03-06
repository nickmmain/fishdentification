import numpy as np
import cv2
import matplotlib.pyplot as plt
from fishes import test_image, nemo
from show import show_all_frames


def spampinatoKernels():
    # 6 scales, 4 orientations. That takes care of ksize and theta.
    # there may be a weakness because I am only using symmetric kernels (i.e. (4,4)); why not oblong kernels ?

    # for sigma, we want to choose a value which will allow the filter to sift out features of the fish.
    # may imply that certain fish (perhaps even certain sizes of fish) will be more distinguishable by these features than others.

    # lambda let's fix it at np.pi/4

    # gamma will control how oblong the kernel is; let's go with a circular kernel (gamma=1)
    angles = [np.pi*i/4 for i in range(1, 5)]
    scales = [2*i for i in range(1, 7)]
    lamda = np.pi/4
    gamma = 1
    kernels = {}

    kernels = [
        [cv2.getGaborKernel((scales[i], scales[i]), 6, angles[j], lamda, gamma)
         for i in range(len(scales))]
        for j in range(len(angles))]

    flat_kernels = [item for sublist in kernels for item in sublist]

    return flat_kernels


kernels = spampinatoKernels()


def showKernels(plots):
    '''Shows the static Gabor kernels used by this project'''

    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    columns = 6
    rows = 4
    for i in range(1, columns*rows + 1):
        img = plots[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


def gaborFeatures(img):
    '''For the image given, this convolves static Gabor kernels with the image given. Mean and Standard deviation are then returned.'''

    gaborFeatures = {}
    gaborFeatures['gaborFeatures'] = {}
    gaborFeatures['gaborFeatures']['mean'] = []
    gaborFeatures['gaborFeatures']['stdDeviation'] = []

    # applying gabor kernels from opencv gives one value for every pixel in the image.
    filteredImgs = [cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    for kernel in kernels]

    # we take the mean and standard deviation for every filtered image:
    for img in filteredImgs:
        gaborFeatures['gaborFeatures']['mean'].append(np.mean(img))
        gaborFeatures['gaborFeatures']['stdDeviation'].append(np.std(img))

    return gaborFeatures


if __name__ == "__main__":
    # img = test_image(True)
    # gaborFeatures(img)

    # kernels = spampinatoKernels()
    # plt.imshow(kernels[4])
    # plt.show()

    # kernels = spampinatoKernels()
    # showKernels(kernels)

    kernels = spampinatoKernels()
    img = nemo(True, True)
    filteredImgs = [cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    for kernel in kernels]
    showKernels(filteredImgs)
