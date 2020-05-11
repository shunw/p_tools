from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import imutils
import sys

'''
this is to show the text message in one reciept with the top-down view. 
'''

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

def doc_scan_trial(fl_name): 
    args = {'image': fl_name}

    # load the image and compute the ratio of the old height to the new height, clone it, and resize it
    image = cv2.imread(args['image'])
    cv2.startWindowThread()
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)

    # convert the image to grayscale, blur it and find edges in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 90) # previous is 75, 200; kindle one 50, 90


    # # to connect the contour
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # for kindle pic 5, 5
    dilated = cv2.dilate(edged, kernel)

    # # show the original image and the edge detected image
    # print ('STEP 1: Edge Detection')
    # cv2.imshow('Image', image)
    # cv2.imshow('Edged', edged)
    # cv2.imshow('dilated', dilated)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # find the controus in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for ind, c in enumerate(cnts): 
        # approximate the contour
        peri = cv2.arcLength(c, True) # <- 计算封闭周长
        approx = cv2.approxPolyDP(c, .02 * peri, True) # <- 计算多边形有多少条边

        # if our approximated contour has four points, then we can assume that we have found our screen
        if  len(approx) == 4: 
            screenCnt = approx
            break
        
        # if all the cnt is checked but no len(approx) == 4, alert warning
        if ind == len(cnts) - 1: 
            print ('no rectangle found, please check the source image. ')
            sys.exit(0)
        
    # # show the contour (outline) of the piece of paper
    # print ('STEP 2: Find contours of paper')
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow('Outline', image)
    
    # cv2.startWindowThread()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    warped = add_sharpen_kernel(warped, False)
    # convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    T = threshold_local(warped, 11, offset = 20, method = 'gaussian') # (warped, 11, offset = 10, method = 'gaussian')
    warped = (warped > T).astype('uint8') * 255 
    # show the original and scanned images
    print ('STEP 3: Apply perspective transform')
    cv2.imshow('Original', imutils.resize(orig, height = 650))
    cv2.imshow('Scanned', imutils.resize(warped, height = 650))

    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__': 
    doc_scan_trial('check_scan_img_2.JPG')
    
    

    