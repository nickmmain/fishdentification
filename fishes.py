import cv2
import os
import re
import sys


# fish dataset from: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/


def test_image(gray=True):
    img = cv2.imread(os.path.join(os.getcwd(), 'data', 'tawachiche.jpg'))
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# def trainingImages(gray=True, trainingPortion=0.7):


def trainingMasks(trainingPortion=0.7, fish='all'):
    dataPath = os.path.join(os.getcwd(), 'data')
    maskFolders = []

    for (dirpath, dirnames, filenames) in os.walk(dataPath):
        for dir in dirnames:
            masks = re.findall("^mask.*", dir)
            if(len(masks) != 0):
                maskFolders.append(os.path.join(dirpath, masks[0]))

    trainingMaskPaths = []
    for maskFolder in maskFolders:
        maskPaths = []
        for (dirpath, dirnames, filenames) in os.walk(maskFolder):
            for picture in filenames:
                maskPaths.append(os.path.join(dirpath, picture))

        trainingMaskPaths.append(maskPaths)

    return trainingMaskPaths


if __name__ == "__main__":
    trainingMasksArr = trainingMasks()
    print("Whatup")
