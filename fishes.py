import cv2
import os


def test_image(gray=True):
    img = cv2.imread(os.path.join(os.getcwd(), 'data', 'tawachiche.jpg'))
    if gray:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img
