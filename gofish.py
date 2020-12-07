import json
import cv2
from fishes import fishes, masks
from features import getFeaturesArray

# testing of the LDA model with 100 pictures of fish
imgPaths = fishes()
features = []
for imgPath in imgPaths:
    img = cv2.imread(imgPath)
    imgFeatures = getFeaturesArray(img)
