'''
purpose: use mouse to draw arrow
how to use: 
    1. put your file name in the line 9
    2. NOTICE: when you draw the line, the image won't show up. 
    3. You are able to draw several lines on the same image. 
'''

import cv2, numpy as np

pic_name = 'image.png'
image = cv2.imread(pic_name)
image_to_show = np.copy(image)

mouse_pressed = False
s_x = s_y = e_x = e_y = -1

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed 
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        # image_to_show = np.copy(image_to_show)
    # elif event == cv2.EVENT_MOUSEMOVE:
    # # this is to use the process image, your mouse to which area
    #     if mouse_pressed:
    #         # image_to_show = np.copy(image_to_show)
    #         # cv2.rectangle(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)
    #         # cv2.arrowedLine(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1, cv2.LINE_AA)
    #         pass
    elif event == cv2.EVENT_LBUTTONUP: 
        
        mouse_pressed = False
        e_x, e_y = x, y
        cv2.arrowedLine(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1, cv2.LINE_AA)


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True: 
    
    cv2.imshow('image', image_to_show)
    cv2.startWindowThread()
    k = cv2.waitKey(1)
    
    # crop
    if k == ord('c'): 
        if s_y > e_y:
            s_y, e_y = e_y, s_y
        if s_x > e_x: 
            s_x, e_x = e_x, s_x
        if e_y - s_y > 1 and e_x - s_x > 0: 
            image = image_to_show[s_y:e_y, s_x: e_x]
            image_to_show = np.copy(image)
    elif k == ord('s'): 
        # save the image 
        cv2.imwrite('image_to_show.png', image_to_show)
    elif k == 27:
        break
cv2.destroyAllWindows()

    


