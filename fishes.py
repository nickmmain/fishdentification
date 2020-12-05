import cv2
import os

# fish dataset from: http://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/RECOG/


def test_image(gray=True):
    img = cv2.imread(os.path.join(os.getcwd(), 'data', 'tawachiche.jpg'))
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def trainingImages(gray=True):
