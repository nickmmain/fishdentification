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


def splitData(data, trainingPortion=0.7):
    '''splits the data into training and testing groups along the given fraction'''
    for fish in data:
        fishDataPaths = allFish[fish]
        trainIndex = floor(trainingPortion*len(fishDataPaths))
        allFish[fish] = {}
        allFish[fish]['train'] = fishDataPaths[:trainIndex]
        allFish[fish]['test'] = fishDataPaths[trainIndex+1:]

    return allFish


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
    trainingMasksArr = fishes()
    print("Whatup")
