import cv2
import os
import re
import sys
import random
from math import floor, ceil

# fish dataset from: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/
dataPath = os.path.join(os.getcwd(), 'data')
random.seed()


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


def fishesAndMasks(trainingPortion, maxFish):
    '''First gets all pictures of fish in directories starting with "fish", then gets corresponding masks in corresponding "mask" folders'''
    fishes = getFishes(maxFish)
    masks = getMasksForFishes(fishes)

    fishes = splitData(fishes, trainingPortion)
    masks = splitData(masks, trainingPortion)

    fishesAndMasks = {}
    fishesAndMasks['fish'] = fishes
    fishesAndMasks['masks'] = masks
    fishesAndMasks['data_dir'] = dataPath

    return fishesAndMasks


def getFishes(maxFish):
    '''returns all files in folders that start with "fish" in the data directory of this project'''
    allFishFolders = getDataFolders("^fish.*")
    for fishFolder in allFishFolders:
        allFishFolders[fishFolder] = getData(
            os.path.join(dataPath, fishFolder), maxFish)

    return allFishFolders


def getMasksForFishes(fishes):
    masks = {}

    for fish in fishes:
        maskForFish = fish.replace('fish', 'mask')
        masks[maskForFish] = []
        for picFileName in fishes[fish]:

            maskFileName = picFileName.replace('fish', 'mask')
            fullMaskPath = (os.path.join(dataPath, maskForFish, maskFileName))

            assert os.path.exists(
                fullMaskPath), "coulnd't find the mask which correponds to "+picFileName+":"+fullMaskPath

            masks[maskForFish].append(maskFileName)

    return masks


def getMasks(maxMasks):
    '''returns all files in folders that start with "mask" in the data directory of this project'''
    allMasks = getDataFolders("^mask.*")
    for maskFolder in allMasks:
        allMasks[maskFolder] = getData(
            os.path.join(dataPath, maskFolder), maxMasks, False)
    return allMasks


def splitData(dataFolders, trainingPortion=0.7):
    '''splits the data into training and testing groups along the given fraction'''
    for dataFolder in dataFolders:
        dataPaths = dataFolders[dataFolder]
        trainUpToIndex = floor(trainingPortion*len(dataPaths))
        dataFolders[dataFolder] = {}
        dataFolders[dataFolder]['train'] = dataPaths[:trainUpToIndex]
        dataFolders[dataFolder]['test'] = dataPaths[trainUpToIndex+1:]

    return dataFolders


def getDataFolders(folderRegex):
    '''returns folders that match folderRegex as a dictionary'''
    dataFolders = {}

    for (dirpath, dirnames, filenames) in os.walk(dataPath):
        for dir in dirnames:
            dataFolderMatchingRe = re.findall(folderRegex, dir)
            if(len(dataFolderMatchingRe) != 0):
                dataFolders[dataFolderMatchingRe[0]] = []

    return dataFolders


def getData(dataFolderPath, maxFiles, randomFiles=True):
    filePaths = []

    for (dirpath, dirnames, filenames) in os.walk(dataFolderPath):

        filesIndices = []
        if(randomFiles):
            filesIndices = random.sample(range(len(filenames)), maxFiles)
        else:
            filesIndices = range(maxFiles)

        for filesIndex in filesIndices:
            filePaths.append(filenames[filesIndex])

    return filePaths


if __name__ == "__main__":
    fish = fishesAndMasks(0.7, 100)
    print("Whatup")
