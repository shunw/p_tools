from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import sys

def add_sharpen_kernel(img, img_path = True): 
    '''
    this is to sharpent the image
    img is the image path or image itself
    img_path is to check if input is image or image path
    '''
    if img_path: 
        image = cv2.imread(img)
    else: 
        image = img.copy()
    # cv2.imshow('original', image)

    sharpen_1 = np.array([[-1, -1, -1], 
                            [-1, 9, -1],
                            [-1, -1, -1]])

    sharpen_2 = np.array([[0, -1, 0], 
                            [-1, 5, -1],
                            [0, -1, 0]])

    sobelY = np.array([[-1, -2, -1], 
                        [0, 0, 0],
                        [1, 2, 1]]) # not work. to sharpen the horizontal
    sharpened = cv2.filter2D(image, -1, sharpen_1)
    # cv2.imshow('sharpened', sharpened)
    # cv2.waitKey(0)
    return sharpened