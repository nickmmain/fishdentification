import cv2
import numpy as np
import itertools


def show_all_frames(imgs, windowName, numImgsWide, numImgsHigh, save=False):
    margin = 20  # Margin between pictures in pixels
    n = numImgsWide*numImgsHigh

    img_h, img_w, img_c = imgs[0].shape

    # Define the margins in x and y directions
    m_x = margin
    m_y = margin

    # Size of the full size image
    mat_x = img_w * numImgsWide + m_x * (numImgsWide - 1)
    mat_y = img_h * numImgsHigh + m_y * (numImgsHigh - 1)

    # Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    imgmatrix = np.zeros((mat_y, mat_x, img_c), np.uint8)
    imgmatrix.fill(255)

    # Prepare an iterable with the right dimensions
    positions = itertools.product(range(numImgsHigh), range(numImgsWide))

    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img

    resized = cv2.resize(imgmatrix, (mat_x//3, mat_y//3),
                         interpolation=cv2.INTER_AREA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, resized)

    if(save):
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        cv2.imwrite(os.path.join(os.getcwd(), "AllFrames.jpg"),
                    resized, compression_params)
