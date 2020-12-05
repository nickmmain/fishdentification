import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    kernels = [
        [cv2.getGaborKernel((scales[i], scales[i]), 6, angles[j], lamda, gamma)
         for i in range(len(scales))]
        for j in range(len(angles))]

    flat_kernels = [item for sublist in kernels for item in sublist]

    return flat_kernels


if __name__ == "__main__":
    kernels = spampinatoKernels()
    plt.imshow(kernels[20])
    plt.show()
