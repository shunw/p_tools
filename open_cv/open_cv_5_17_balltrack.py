from collections import deque
from imutils.video import VideoStream
import numpy as np
import cv2
import imutils
import time

buf = 64
img_name = 'test_1.jpeg'
args = {'img': img_name, 'buffer': buf}

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
# pts = deque(maxlen= args['buffer'])
img = cv2.imread(args['img'])
blurred = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, greenLower, greenUpper)
# mask = cv2.erode(mask, None, iterations= 2)
# mask = cv2.dilate(mask, None, iterations= 2)
cv2.imshow('org', img)
cv2.imshow('hsv', hsv)
cv2.imshow('mask', mask)
cv2.startWindowThread()
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)