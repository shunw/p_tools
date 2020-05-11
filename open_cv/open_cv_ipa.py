import numpy as np
import cv2
import imutils
from pyzbar import pyzbar

'''
this script is to make measure the image placement. several data should got as following: 
    1. ABCDEFGH
    2. TLX, TLY, TRX, TRY, BLX, BLY, BRX, BRY (get the atual)

methods: 
    option 1 --- use the camera top-down view
    problem: 
        ? if the ratio is different, how to deal
            - use the four ratio
            - this is why the fixture use the four camera just above the image dots, there is no need to deal with the distort
        ? what about the distance calibration between tl vs tr; tl vs the bl
            - maybe this is the reason the TLX, TLY, etc, is not correct. 
    1. calibration in four corner with fiducial dots (white dots with black background) ==> get the "pixels per metric" ratio
    2. get the image photo
    3. deal with python

    option 2 --- use scanner to get the picture
    problem: 
        ? how to determine the edge of the printouts since the scan image is scanned based on the media size. 

    option 3 --- use cellphone to take picture and measure
    problem: 
        ? may skew
        ? not sure the accuracy

steps: 
    0. read the barcode information 
    1. find the outline (check the shape to see if this is standard media size)
        1.1. consider scan image may skewed
    2. find four dots outlines
    3. find the dots center
    4. calculate the center to edges, xy
    5. calculate the center to the 0, 0, xy(tlx, tly, etc)


verification: 
    0. check the outline (rectangle) width and height
    1. check other skewd scan image

current situation: 
    1. practice the example, measure something, and check the accuracy
'''
def show_cv2_img_mac(): 
    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def resized_image_smaller(img, height = 600): 
    output = imutils.resize(img, height = height)
    cv2.imshow('image', output)
    show_cv2_img_mac()
    


def find_four_dots(image): 
    '''
    this is to find four dots at the four corners
    input: image array
    return: four dots center in numpy array format
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # background turn black, highlighted the dots and the barcode
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    dotsCnts = []
    centers = []
    
    for c in cnts: 

        # compute the bouding box of the contour, then use the bouding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and have an aspect ratio approximately equal to 1
        if ar >= .9 and ar <= 1.1: # this is to find the circle, according to it's w and h ratio
            dotsCnts.append(c)
    dotsCnts = sorted(dotsCnts, key = cv2.contourArea, reverse = True)[:4] # use the contour area to find the filled dots position; could not use this directly due to the barcode area is also solid.

    for d in dotsCnts: # this is to find the centers of dots
        ((x, y), radius)= cv2.minEnclosingCircle(d)
        M = cv2.moments(d)
        center = (int(M['m10']/ M['m00']), int(M['m01']/ M['m00']))
        centers.append(center) # according to the sort, the biggest area is bottom right
    
    # # following is to circle the dots
    # output = image.copy()
    # for ind, q in enumerate(dotsCnts):  
    #     (x, y, w, h) = cv2.boundingRect(q)
    #     cx, cy = centers[ind]
    #     cv2.circle(output, (cx, cy), int(w/2.0), (0, 255, 0), 2)
    
    # # output = imutils.resize(output, height = 500) # to resize the image height
    # cv2.imshow('output', output)
    # show_cv2_img_mac()
    return centers

def get_dot_2_edge_distance(): 
    '''
    '''


def read_barcode_info(image): 
    '''
    this is to read through barcode information
    input: image array
    return: barcode information
    '''
    barcodes = pyzbar.decode(image)

    # loop over the detected barcodes
    for barcode in barcodes: 

        # extract the bouding box location of the barcode and draw the bouding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # the barcode data is a bytes object so if we want to draw it on our output image we need to convert it to a string first
        barcodeData = barcode.data.decode('utf-8')
        barcodeType = barcode.type

        # draw the barcode data and barcode type on the image
        text = '{} ({})'.format(barcodeData, barcodeType)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    
    # cv2.imshow('image', image)
    # show_cv2_img_mac()
    # resized_image_smaller(image)
    project_id, dev_id, page_number = barcodeData.split(':')[0].split('.')
    print (project_id, dev_id, page_number)
    

if __name__ == '__main__': 
    img = cv2.imread('wscan3027.jpg')
    a, b, c, d =  find_four_dots(img)
    # read_barcode_info(img)
    print (a, b, c, d)
    