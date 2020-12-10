import cv2
import os
import re
import sys
from math import floor, ceil

# fish dataset from: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/
dataPath = os.path.join(os.getcwd(), 'data')


def test_image(gray=True):
    img = cv2.imread(os.path.join(os.getcwd(), 'data', 'tawachiche.jpg'))
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def nemo(mask=True, gray=True):
    '''A good example from the dataset of fish04'''
    img = cv2.imread(os.path.join(os.getcwd(), 'data',
                                  'fish_04', 'fish_000004859599_08020.png'))
    if mask:
        mask = cv2.imread(os.path.join(os.getcwd(), 'data',
                                       'mask_04', 'mask_000004859599_08020.png'))
        grayMask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        img = cv2.bitwise_and(img, img, mask=grayMask)
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def fishesAndMasks(split=True):
    allFish = fishes(split)
    allMasks = masks(split)
    fishes_masks = {}
    fishes_masks['fish'] = allFish
    fishes_masks['masks'] = allMasks

    return fishes_masks


def fishes(split=True):
    '''returns all files in folders that start with "fish" in the data directory of this project'''
    allFish = getData("^fish.*")
    if(split):
        return splitData(allFish, 0.7)
    return allFish


def masks(split=True):
    '''returns all files in folders that start with "mask" in the data directory of this project'''
    allMasks = getData("^mask.*")
    if(split):
        return splitData(allMasks, 0.7)
    return allMasks


def splitData(dataFolders, trainingPortion=0.7):
    '''splits the data into training and testing groups along the given fraction'''
    for dataFolder in dataFolders:
        fishDataPaths = dataFolders[dataFolder]
        trainIndex = floor(trainingPortion*len(fishDataPaths))
        dataFolders[dataFolder] = {}
        dataFolders[dataFolder]['train'] = fishDataPaths[:trainIndex]
        dataFolders[dataFolder]['test'] = fishDataPaths[trainIndex+1:]

    return dataFolders


def getData(folderRegex):
    '''returns all files in folders that match folderRegex as a dictionary'''
    dataFolders = {}

    for (dirpath, dirnames, filenames) in os.walk(dataPath):
        for dir in dirnames:
            dataFolderMatchingRe = re.findall(folderRegex, dir)
            if(len(dataFolderMatchingRe) != 0):
                dataFolders[dataFolderMatchingRe[0]] = []

    for dataFolder in dataFolders:
        dataFolderPath = os.path.join(dataPath, dataFolder)
        for (dirpath, dirnames, filenames) in os.walk(dataFolderPath):
            for fyle in filenames:
                dataFolders[dataFolder].append(os.path.join(dirpath, fyle))

    return dataFolders


if __name__ == "__main__":
    trainingMasksArr = fishesAndMasks()
    print("Whatup")
