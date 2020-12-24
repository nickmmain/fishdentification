import cv2
import numpy as np


def checkForMultipleContours(mask, contours, maxContours=1):
    '''Check the contours drawn to make sure there is only 1. 
    If there is more than 1, highlight and display them for debugging.'''

    if(len(contours) > maxContours):
        # if the masks are properly done, with RETR_EXTERNAL, this code shouldn't be executed.
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        draw_on_these = cv2.cvtColor(np.copy(mask), cv2.COLOR_GRAY2BGR)

        for i in range(len(contours)):
            draw_on_these = cv2.drawContours(
                draw_on_these, contours[i], -1, colors[i % 3], 3)

        window_name = 'more than 1 contour !'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, draw_on_these)
        cv2.waitKey()
