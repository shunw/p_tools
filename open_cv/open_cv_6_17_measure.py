'''
measure objects on one picture
'''

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

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

def midpoint(ptA, ptB): 
    # compute the midpoint between two sets of (x, y) - coordinates
    return ((ptA[0] + ptB[0]) * .5, (ptA[1] + ptB[1]) * .5)

def mac_show_cv2(): 
    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def cnts_2_ratio(cnts, ratio, conts_more_than_1 = True): 
    '''
    cnts judge by the original images, this purpose is to make the cnts fit to the ratio which will shown
    input: 
        cnts, type may be the list; 
        ratio, image height/ want to shown height
        conts_more_than_1 -- deal with the single cnts situation for debug purpose
    '''
    cnts_resize = cnts.copy()
    if conts_more_than_1: 
        for cnts_ind, cnts_value in enumerate(cnts_resize): 
            # print (len(c))
            for row_ind, row_value in enumerate(cnts_value):
                cnts_value[row_ind] = [[int(j/ratio) for i in row_value for j in i]]
        
    else: 
        for row_ind, row_value in enumerate(cnts_resize):
            cnts_resize[row_ind] = [[int(j/ratio) for i in row_value for j in i]]
    return cnts_resize

def similar_length(a, b, threshold): 
    '''
    compare two length is similar length or not
    return: True/ False
    '''
    if min(a, b)/(max(a, b) * 1.0) > threshold: 
        return True
    else: 
        return False


def clear_cnt(cnts, size_diff_threshold = .97): 
    '''
    input: cnt list <- sorted one, which is sorted by the contour area;
        size_diff_threshold: bigger than this, will consider to exclude this image
    purpose: 
        1. to clear the contours, exclude the similar position, by checking the four corner position. 
        2. need to limit the cnt for the rectangle, since both the outline and the four corner square are 4 edged
    '''
    remain_ind_list = []
    filtered_cnt = []
    
    for c in cnts: 
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .02 * peri, True) # previous .04

        if len(approx) == 4:
            filtered_cnt.append(c)

    for ind, fc in enumerate(filtered_cnt):    
        if ind == 0: 
            pre_pos = cv2.boundingRect(fc)
        else: 
            current_pos = cv2.boundingRect(fc)
            temp_true = 0
            for a, b in zip(current_pos, pre_pos): 
                temp_true += similar_length(a, b, size_diff_threshold)
            if temp_true <= 3: 
                remain_ind_list.append(ind)
            pre_pos = current_pos
        # print (x, y, w, h)
    
    return [filtered_cnt[i] for i in remain_ind_list]
    
def name_contour(suitable_shown_image, resized_cnts, conts_more_than_1= True): 
    '''
    purpose is to named contour, make contour on the image. help to check which one is missing
    input: resized_cnts is list; 
            image is np array shown
            conts_more_than_1 -- deal with the single cnts situation for debug purpose
    output: a process, no actual output
    '''
    if conts_more_than_1:
        for ind, value in enumerate(resized_cnts): 
            cv2.drawContours(suitable_shown_image, [value], -1, (0, 255, 0), 2)
            cv2.putText(suitable_shown_image, '{}'.format(ind), (value[0][0][0], value[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    else: 
        cv2.drawContours(suitable_shown_image, resized_cnts, -1, (0, 255, 0), 2)
        cv2.putText(suitable_shown_image, '{}'.format(0), (resized_cnts[0][0][0], resized_cnts[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
def get_out_rect_approx_n_sorted_cnts(cnts, slicer = 0, conts_more_than_1 = False): 
    '''
    input: 
        cnts --- list of cnts, maybe several, maybe just one
        slicer --- one number or slice to get biggest area
        conts_more_than_1 --- check if input conts is more than 1 or just 1

    output: 
        cnts --- sorted and sliced
        out_rect_approx --- output the four dots of biggest outline

    purpose: 
        make sure even 1 cnts also could work smoothly, when to show the conts on that image
    '''
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[slicer] #[1:10] 

    
    if  conts_more_than_1: 
        # remove the largest one; similar position cnt; just get the 5 contours
        cnts_before_clear = cnts.copy()
        cnts = clear_cnt(cnts_before_clear)[:5]

        # get the four dots of the rectangle
        peri = cv2.arcLength(cnts[0], True)
        out_rect_approx = cv2.approxPolyDP(cnts[0], .02 * peri, True)

    else: 
        peri = cv2.arcLength(cnts, True)
        out_rect_approx = cv2.approxPolyDP(cnts, .02 * peri, True)
    
    return cnts, out_rect_approx

def pre_deal_4_measure(img_path): 
    '''
    input: image path
    output: 
        image --- rectangle with four squared calibration area 

    purpose: to make the image to square smooth one, in order for the further measurement
    
    help to make the contour: 
        - ! make sure the contour is obvious, HB pencil is not a good choice
        - add Gaussianblur
        - add getStructuringElement
    '''
    image = cv2.imread(img_path)

    # due to the HB pencil is too light, need sharpen deal
    # sharpened_image = add_sharpen_kernel(image, img_path = False)

    # deal in gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 20, 100) # 150 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # for kindle pic 5, 5
    edged = cv2.dilate(edged, kernel)


    # find contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print (len(cnts))
    cnts = imutils.grab_contours(cnts) # get the cnts list information, the 1st item of the previous line
    
    # deal with cnts
    slicer = slice(1, 10, None)
    slicer = 2 

    conts_more_than_1 = False
    if type(slicer) is slice: 
        conts_more_than_1 = True

    cnts, out_rect_approx = get_out_rect_approx_n_sorted_cnts(cnts, slicer, conts_more_than_1)


    # get the top-bottom image and calculate the length
    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(image, out_rect_approx.reshape(4, 2))
    
    # # show image part => for debug purpose
    # suitable_shown_image = imutils.resize(warped, height = 600)
    # # ratio = image.shape[0]/600
    # # resized_cnts = cnts_2_ratio(cnts, ratio, conts_more_than_1= conts_more_than_1)
    
    # # name_contour(suitable_shown_image, resized_cnts, conts_more_than_1= conts_more_than_1)
    # cv2.imshow('image.png', suitable_shown_image)
    # mac_show_cv2()
    return warped

def mark_cnts_center(cnts): 
    '''
    input: cnts
    purpose: to mark/ return the center of each cnts
    '''

    for c in cnts: 

        # ==================== USE MOMENT ====================
        # peri = cv2.arcLength(cnts[0], True)
        
        # # return the four corner points of the rectangle
        # out_rect_approx = cv2.approxPolyDP(cnts[0], .02 * peri, True)
        # print (out_rect_approx.reshape(4, 2))

        # M = cv2.moment(c)
        # cX = int(M['m10']/ M['m00'])
        # cY = int(M['m01']/ M['m00'])
        # print (cX, cY)

        # ==================== END ====================
        # print (c)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        print (box)

        # break

def make_up_edged(img_array, broken_pixel_thresh = 5):
    '''
    this should be used after rectangle area defined
    purpose: 
        - if the edge data for the four corner square shown as 0, change to 255. make sure all the lines for the fidual square are continuous
        - change the four corner square, suppose the square line is smaller than total length 1/8
    input: 
        image_array --- all the image data array
        pixel --- how many pixel to makeup the broken boarder
    output: 
        fixed image data array
    current:
        just make up for the bottom left, bottom line
    '''
    m, n = img_array.shape
    # check_length = 250 # this is to check the four corner square [for debug]
    line_thick = 5 # this is to check the thick of the line, to see if the whole line are broken. Or just thin part broken.

    # check the col data are same or not & check the col data == 0
    line_bottom = img_array[m - line_thick:, :] # bottom edge
    line_top = img_array[: line_thick, :] # top edge
    line_left = img_array[:, : line_thick] # left vertical edge
    line_right = img_array[:, m - line_thick] # right vertical edge
    
    # check the top and bottom line 
    for ind, l in enumerate([line_top, line_bottom]): 
        col_ind = np.all(l == np.zeros((1, n)), axis = 0)
        broken_length = col_ind.sum()
        if broken_length > 0 and broken_length <= broken_pixel_thresh: 
            # print (1)
            if ind == 0:
                l[0, col_ind] = np.array([[255] * col_ind.sum()])
            else: 
                l[-1, col_ind] = np.array([[255] * col_ind.sum()])
    
    # check the left and right line
    for ind, l in enumerate([line_left, line_right]): 
        l_equal_0 = (l == np.zeros((m, 1))).sum(axis = 1)
        row_ind = l_equal_0 == line_thick
        
        broken_length = row_ind.sum()
        if broken_length > 0 and broken_length <= broken_pixel_thresh: 
            # print (1)
            if ind == 0:
                l[row_ind, 0] = np.array([[255] * row_ind.sum()])
            else: 
                l[row_ind, -1] = np.array([[255] * row_ind.sum()])
    # print (line[0, :])
    # print (type(line))
    # _check_broken_line(line, pixel)
    # cv2.imshow('line', line)
    # mac_show_cv2()
    
    return img_array
    


def get_four_calibration_square(pre_image): 
    '''
    input: 
        pre_image --- is pre-deal, and shown in the top-down view
    output: 
        x, y, w, h of the four corner calibration square
        need to figure out the tl, tr, bl, br
    '''
    
    # deal in gray image
    gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.medianBlur(gray, 3) # 5

    # gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(gray, 50, 150) # 50, 200

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 5
    edged = cv2.dilate(edged, kernel)

    # =========== debug process: check edge image ==================
    part_image = edged[:250, 2179 - 250:].copy() # 2179/ 2347
    
    cnts_part = cv2.findContours(part_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_SIMPLE
    cnts_part = imutils.grab_contours(cnts_part)
    cnts_part = sorted(cnts_part, key = cv2.contourArea, reverse = True).copy()
    # for c in cnts_part: 
    #     print (cv2.contourArea(c))
    print (part_image.shape)
    # print (cnts_part[0])
    name_contour(part_image, cnts_part)
    cv2.imshow('part_image', part_image)
    mac_show_cv2()
    # =========== debug end: check each corner square ==================

    # this is to make up the edge by hand
    edged = make_up_edged(edged, 10)

    # find contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_SIMPLE
    cnts = imutils.grab_contours(cnts) # get the cnts list information, the 1st item of the previous line
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] 
    # for c in cnts: 
    #     print (cv2.contourArea(c))
    cnts_before_clear = cnts.copy()
    cnts = clear_cnt(cnts_before_clear)[:10]
    cnts_1, out_rect_approx_1 = get_out_rect_approx_n_sorted_cnts(cnts)
    # print (cnts_1)
'''
measure objects on one picture
'''

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local

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

def midpoint(ptA, ptB): 
    # compute the midpoint between two sets of (x, y) - coordinates
    return ((ptA[0] + ptB[0]) * .5, (ptA[1] + ptB[1]) * .5)

def mac_show_cv2(): 
    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def cnts_2_ratio(cnts, ratio, conts_more_than_1 = True): 
    '''
    cnts judge by the original images, this purpose is to make the cnts fit to the ratio which will shown
    input: 
        cnts, type may be the list; 
        ratio, image height/ want to shown height
        conts_more_than_1 -- deal with the single cnts situation for debug purpose
    '''
    cnts_resize = cnts.copy()
    if conts_more_than_1: 
        for cnts_ind, cnts_value in enumerate(cnts_resize): 
            # print (len(c))
            for row_ind, row_value in enumerate(cnts_value):
                cnts_value[row_ind] = [[int(j/ratio) for i in row_value for j in i]]
        
    else: 
        for row_ind, row_value in enumerate(cnts_resize):
            cnts_resize[row_ind] = [[int(j/ratio) for i in row_value for j in i]]
    return cnts_resize

def similar_length(a, b, threshold): 
    '''
    compare two length is similar length or not
    return: True/ False
    '''
    if min(a, b)/(max(a, b) * 1.0) > threshold: 
        return True
    else: 
        return False


def clear_cnt(cnts, size_diff_threshold = .97): 
    '''
    input: cnt list <- sorted one, which is sorted by the contour area;
        size_diff_threshold: bigger than this, will consider to exclude this image
    purpose: 
        1. to clear the contours, exclude the similar position, by checking the four corner position. 
        2. need to limit the cnt for the rectangle, since both the outline and the four corner square are 4 edged
    '''
    remain_ind_list = []
    filtered_cnt = []
    
    for c in cnts: 
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, .02 * peri, True) # previous .04

        if len(approx) == 4:
            filtered_cnt.append(c)

    for ind, fc in enumerate(filtered_cnt):    
        if ind == 0: 
            pre_pos = cv2.boundingRect(fc)
        else: 
            current_pos = cv2.boundingRect(fc)
            temp_true = 0
            for a, b in zip(current_pos, pre_pos): 
                temp_true += similar_length(a, b, size_diff_threshold)
            if temp_true <= 3: 
                remain_ind_list.append(ind)
            pre_pos = current_pos
        # print (x, y, w, h)
    
    return [filtered_cnt[i] for i in remain_ind_list]
    
def name_contour(suitable_shown_image, resized_cnts, conts_more_than_1= True): 
    '''
    purpose is to named contour, make contour on the image. help to check which one is missing
    input: resized_cnts is list; 
            image is np array shown
            conts_more_than_1 -- deal with the single cnts situation for debug purpose
    output: a process, no actual output
    '''
    if conts_more_than_1:
        for ind, value in enumerate(resized_cnts): 
            cv2.drawContours(suitable_shown_image, [value], -1, (0, 255, 0), 2)
            cv2.putText(suitable_shown_image, '{}'.format(ind), (value[0][0][0], value[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    else: 
        cv2.drawContours(suitable_shown_image, resized_cnts, -1, (0, 255, 0), 2)
        cv2.putText(suitable_shown_image, '{}'.format(0), (resized_cnts[0][0][0], resized_cnts[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
def get_out_rect_approx_n_sorted_cnts(cnts, slicer = 0, conts_more_than_1 = False): 
    '''
    input: 
        cnts --- list of cnts, maybe several, maybe just one
        slicer --- one number or slice to get biggest area
        conts_more_than_1 --- check if input conts is more than 1 or just 1

    output: 
        cnts --- sorted and sliced
        out_rect_approx --- output the four dots of biggest outline

    purpose: 
        make sure even 1 cnts also could work smoothly, when to show the conts on that image
    '''
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[slicer] #[1:10] 

    
    if  conts_more_than_1: 
        # remove the largest one; similar position cnt; just get the 5 contours
        cnts_before_clear = cnts.copy()
        cnts = clear_cnt(cnts_before_clear)[:5]

        # get the four dots of the rectangle
        peri = cv2.arcLength(cnts[0], True)
        out_rect_approx = cv2.approxPolyDP(cnts[0], .02 * peri, True)

    else: 
        peri = cv2.arcLength(cnts, True)
        out_rect_approx = cv2.approxPolyDP(cnts, .02 * peri, True)
    
    return cnts, out_rect_approx

def pre_deal_4_measure(img_path): 
    '''
    input: image path
    output: 
        image --- rectangle with four squared calibration area 

    purpose: to make the image to square smooth one, in order for the further measurement
    
    help to make the contour: 
        - ! make sure the contour is obvious, HB pencil is not a good choice
        - add Gaussianblur
        - add getStructuringElement
    '''
    image = cv2.imread(img_path)

    # due to the HB pencil is too light, need sharpen deal
    # sharpened_image = add_sharpen_kernel(image, img_path = False)

    # deal in gray image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 20, 100) # 150 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # for kindle pic 5, 5
    edged = cv2.dilate(edged, kernel)


    # find contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print (len(cnts))
    cnts = imutils.grab_contours(cnts) # get the cnts list information, the 1st item of the previous line
    
    # deal with cnts
    slicer = slice(1, 10, None)
    slicer = 2 

    conts_more_than_1 = False
    if type(slicer) is slice: 
        conts_more_than_1 = True

    cnts, out_rect_approx = get_out_rect_approx_n_sorted_cnts(cnts, slicer, conts_more_than_1)


    # get the top-bottom image and calculate the length
    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(image, out_rect_approx.reshape(4, 2))
    
    # # show image part => for debug purpose
    # suitable_shown_image = imutils.resize(warped, height = 600)
    # # ratio = image.shape[0]/600
    # # resized_cnts = cnts_2_ratio(cnts, ratio, conts_more_than_1= conts_more_than_1)
    
    # # name_contour(suitable_shown_image, resized_cnts, conts_more_than_1= conts_more_than_1)
    # cv2.imshow('image.png', suitable_shown_image)
    # mac_show_cv2()
    return warped

def mark_cnts_center(cnts): 
    '''
    input: cnts
    purpose: to mark/ return the center of each cnts
    '''

    for c in cnts: 

        # ==================== USE MOMENT ====================
        # peri = cv2.arcLength(cnts[0], True)
        
        # # return the four corner points of the rectangle
        # out_rect_approx = cv2.approxPolyDP(cnts[0], .02 * peri, True)
        # print (out_rect_approx.reshape(4, 2))

        # M = cv2.moment(c)
        # cX = int(M['m10']/ M['m00'])
        # cY = int(M['m01']/ M['m00'])
        # print (cX, cY)

        # ==================== END ====================
        # print (c)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        print (box)

        # break

def make_up_edged(img_array, broken_pixel_thresh = 5):
    '''
    this should be used after rectangle area defined
    purpose: 
        - if the edge data for the four corner square shown as 0, change to 255. make sure all the lines for the fidual square are continuous
        - change the four corner square, suppose the square line is smaller than total length 1/8
    input: 
        image_array --- all the image data array
        pixel --- how many pixel to makeup the broken boarder
    output: 
        fixed image data array
    current:
        just make up for the bottom left, bottom line
    '''
    m, n = img_array.shape
    # check_length = 250 # this is to check the four corner square [for debug]
    line_thick = 5 # this is to check the thick of the line, to see if the whole line are broken. Or just thin part broken.

    # check the col data are same or not & check the col data == 0
    line_bottom = img_array[m - line_thick:, :] # bottom edge
    line_top = img_array[: line_thick, :] # top edge
    line_left = img_array[:, : line_thick] # left vertical edge
    line_right = img_array[:, m - line_thick] # right vertical edge
    
    # check the top and bottom line 
    for ind, l in enumerate([line_top, line_bottom]): 
        col_ind = np.all(l == np.zeros((1, n)), axis = 0)
        broken_length = col_ind.sum()
        if broken_length > 0 and broken_length <= broken_pixel_thresh: 
            # print (1)
            if ind == 0:
                l[0, col_ind] = np.array([[255] * col_ind.sum()])
            else: 
                l[-1, col_ind] = np.array([[255] * col_ind.sum()])
    
    # check the left and right line
    for ind, l in enumerate([line_left, line_right]): 
        l_equal_0 = (l == np.zeros((m, 1))).sum(axis = 1)
        row_ind = l_equal_0 == line_thick
        
        broken_length = row_ind.sum()
        if broken_length > 0 and broken_length <= broken_pixel_thresh: 
            # print (1)
            if ind == 0:
                l[row_ind, 0] = np.array([[255] * row_ind.sum()])
            else: 
                l[row_ind, -1] = np.array([[255] * row_ind.sum()])
    # print (line[0, :])
    # print (type(line))
    # _check_broken_line(line, pixel)
    # cv2.imshow('line', line)
    # mac_show_cv2()
    
    return img_array
    


def get_four_calibration_square(pre_image): 
    '''
    input: 
        pre_image --- is pre-deal, and shown in the top-down view
    output: 
        x, y, w, h of the four corner calibration square
        need to figure out the tl, tr, bl, br
    '''
    
    # deal in gray image
    gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.medianBlur(gray, 3) # 5

    # gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(gray, 50, 150) # 50, 200

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # 5
    edged = cv2.dilate(edged, kernel)
    # cv2.imshow('edged', edged)
    # mac_show_cv2()
    # =========== debug process: check edge image ==================
    part_image = edged[:250, 2179 - 250:].copy() # 2179/ 2347
    part_image_org = pre_image[:250, 2179 - 250:].copy() # 2179/ 2347
    
    # cnts_part = cv2.findContours(part_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_SIMPLE
    contours, hier_ = cv2.findContours(part_image.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    cnts_part = imutils.grab_contours(contours)
    cnts_part = sorted(cnts_part, key = cv2.contourArea, reverse = True).copy()
    # for c in cnts_part: 
    #     print (cv2.contourArea(c))
    
    cnts_1, out_rect_approx_1 = get_out_rect_approx_n_sorted_cnts(cnts_part)
    for dot in out_rect_approx_1: 
        print (tuple(dot[0]))
        cv2.circle(part_image_org,tuple(dot[0]), 5, (0,255,0), cv2.FILLED)
    cv2.imshow('part_image', part_image_org)
    mac_show_cv2()
    
    # box = cv2.minAreaRect(out_rect_approx_1)
    # box = cv2.cv.Boxpoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    # box = np.array(box, dtype = 'int')

    # box = perspective.order_points(box)

    # for dot in box: 
    #     print (dot)
    #     cv2.circle(part_image_org,tuple(dot), 5, (0,255,0), cv2.FILLED)
    # cv2.imshow('part_image', part_image_org)
    # mac_show_cv2()
    
    
    # print ('box')
    # print (box)

    print (out_rect_approx_1)
    
    print (part_image.shape)
    # print (cnts_part[0])
    name_contour(part_image, cnts_part)
    
    cv2.imshow('part_image', part_image)
    mac_show_cv2()
    
    # =========== debug end: check each corner square ==================

    # this is to make up the edge by hand
    edged = make_up_edged(edged, 10)

    # find contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_SIMPLE
    cnts = imutils.grab_contours(cnts) # get the cnts list information, the 1st item of the previous line
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10] 
    # for c in cnts: 
    #     print (cv2.contourArea(c))
    cnts_before_clear = cnts.copy()
    cnts = clear_cnt(cnts_before_clear)[:10]
    
    # print (len(cnts))

    # # check contour image   
    # =========== debug process: check each corner square ==================
    # slicer = slice(1, 10, None)
    # slicer = 4

    # conts_more_than_1 = False
    # if type(slicer) is slice: 
    #     conts_more_than_1 = True
    
    # cnts, out_rect_approx = get_out_rect_approx_n_sorted_cnts(cnts, slicer, conts_more_than_1)
    
    # cnts_2b_check = cnts.copy()
    # # cnts_2b_check = out_rect_approx.copy()
    
    # =========== debug process end ==================

    # # find contour center
    # mark_cnts_center(cnts.copy())

    # # shown in image
    suitable_shown_image = imutils.resize(pre_image, height = 600)
    ratio = pre_image.shape[0]/600
    # print ('ratio is {}'.format(ratio))
    resized_cnts = cnts_2_ratio(cnts, ratio) # conts_more_than_1= conts_more_than_1
    
    name_contour(suitable_shown_image, resized_cnts) # , conts_more_than_1 = conts_more_than_1
    cv2.imshow('image.png', suitable_shown_image)
    # cv2.imwrite('image.png', suitable_shown_image)
    mac_show_cv2()


def measure_objects_wh(img_path, reference_width): 
    '''
    to measure the object width and height
    '''
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations = 1)
    edged = cv2.erode(edge, None, iterations = 1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the pixels per metric calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 100: 
            continue
        
        # compute the rotated bounding box of the contour
        orig = image.copy()
        box = cv2.minAreaRect(c)
        box = cv2.cv.Boxpoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype = 'int')

        # order the points in the contour such that they appear in the top-left, top-right, bottom-right, and bottom-left order, then draw the outline of the rotated bounding box
        box = perspective.order_points(box)
        cv2.drawContours(orig, [box.astype('int')], -1, (0, 255, 0), 2)

        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

        # unpack the ordered bounding box, then compute the midpoint between the top-left and top-right coordinates, followed by the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-right and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric (in this case, inches)
        if pixelsPerMetric is None: 
            pixelsPerMetric = dB/ reference_width
        
        # compute the size of the object
        dimA = dA / pixelsPerMetric
        dimB = dB / pixelsPerMetric

        # draw the object size on the image
        cv2.putText(orig, '{:.1f}in'.format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)

        cv2.putText(orig, '{:.1f}in'.format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)

        cv2.imshow('Image', orig)
        cv2.startWindowThread()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        

if __name__ == '__main__': 
    image_path = 'square_line_2_obj.JPG'
    # image_path = 'square_line_not_fill.JPG'
    calibration_item_width = 10 # unit is mm, => 10mm
    args = {'image': image_path, 'width': calibration_item_width}
    
    warped_rec = pre_deal_4_measure(args['image'])
    get_four_calibration_square(warped_rec)
    '''
    failed to download/ 1 use the sample image
        2 take pic by myself with round object
    trying this / 3 take pic by myself with square/ rectangle object, with top-down view
        4 take pic with square/ rectangle object, in some angle
        5 take pic with renference in four corner 
    '''
    