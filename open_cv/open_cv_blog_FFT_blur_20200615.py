from pyimagesearch.blur_detector import detect_blur_fft
import numpy as np
import imutils
import cv2


def mac_show_cv2(): 
    cv2.startWindowThread()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def FFT_blur(img, thresh = 10, vis = False, test = False): 
    '''
    thresh: blur or not judge level
    vis: 
    test: for testing purpose, we can progressively blur our input image and conduct FFT-based blur detection on each example. This flag indicates whether to test or not.
    '''
    
    # load the image, resize, convert to grayscale
    orig = cv2.imread(img)
    orig = imutils.resize(orig, width = 500)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    # apply blur one by FFT
    (mean, blurry) = detect_blur_fft(gray, size = 60) # size, thresh, vis
    
    # draw on the image, indicating where or not it is blurry
    image = np.dstack([gray] * 3)
    color = (0, 0, 255) if blurry else (0, 255, 0)
    text = 'Blurry {:.4f}' if blurry else 'Not Blurry ({:.4f})'
    text = text.format(mean)

    cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)

    print ('[INFO] {}'.format(text))

    # show the output image
    cv2.imshow('Output', image)
    mac_show_cv2()

    # check to see  if are going to test our FFT blurriness detector using various size of a Gaussian Kernel
    if test: 
        # loop over various blur radii
        for radius in range(1, 30, 2):
            # clone the original grayscale image
            image = gray.copy()

            # check to see if the kernel radius is greater than zero
            if radius > 0:
                # blur the input image by the supplied radius using a Gaussian kernel
                image = cv2.GaussianBlur(image, (radius, radius), 0)

                # apply our blur detctor using the FFT
                (mean, blurry) = detect_blur_fft(image, size = 60, thresh = thresh )

if __name__ == '__main__':
    image = 'check_scan_img.JPG'
    FFT_blur(image)