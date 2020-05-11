'''
this is to find filled circles
step 1: detect the sheet image
step 2: apply perspective transform to extract top-down, birds-eye-view of the exam
step 3: extract the set of bubbles from the transformed sheet
step 4: sort the questions/ bubbles into rows
step 5: determine the marked answer for each row. 
step 6: lookup the correct answer 
step 7: repeat for all the questions
'''


import cv2
import imutils

def extract_bubbles(warped):
    '''
    step 3
    input: warped, np array data in gray, which is top-down view of that image
    '''
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the thrsholded image, then initialize the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # loop over the contours
    for c in cnts: 

        # compute the bouding box of the contour, then use the bouding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and have an aspect ratio approximately equal to 1
        if w >= 20 and h >= 20 and ar >= .9 and ar <= 1.1: 
            questionCnts.append(c)