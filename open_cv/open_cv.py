import cv2
import os
import numpy as np
import glob
from imutils.video import FPS
from imutils.video import FileVideoStream
from imutils.video import VideoStream
import imutils
from threading import Thread
from queue import Queue
import time

class CropLayer(object): 
    def __init__(self, params, blobs): 
        # initialize our starting and ending (x, y) - coordinates of the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0
    
    def getMemoryShapes(self, inputs): 
        # the crop layer will receive two inputs -- we need to crop the first input blob to match the shape of the second one, keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we will preform the actual crop during the forward pass)
        return [[batchSize, numChannels, H, W]]
    
    def forward(self, inputs): 
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY, self.startX: self.endX]]
class FileVideoStream: 
    def __init__(self, path, queueSize = 128): 
        # initalize the file video stream along with the boolean used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from the video file
        self.Q = Queue(maxsize = queueSize)
    
    def start(self):
        # start a thread to read frames from the video stream
        t = Thread(target = self.update, args = ())
        t.daemon = True
        t.start()
        return self

    def update(self): 

        # keep looping infinitely
        while True: 
            # if the thread indicator variable is set, stop the thread
            if self.stopped: 
                return 

            # otherwise, ensure the que has room in it
            if not self.Q.full(): 

                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the grabbed boolean is False, then we have reached the end of the video file
                if not grabbed:
                    self.stop()
                    return 
                
                # add the frame to the queue
                self.Q.put(frame)
    def read(self): 
        # return next frame in the queue
        return self.Q.get()
    
    def more(self): 
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
    
    def stop(self): 
        # indicate that the thread should be stopped
        self.stopped = True
def auto_canny(image, sigma = .33): 
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed media
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(image, lower, upper)

    return edged

def canny_no_parameter_compare(img): 
    '''
    deploy the canny with no parameter and compare with others
    img: is the image file name and root
    '''
    args = {'image': img}

    image = cv2.imread(args['image'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    wide = cv2.Canny(gray, 20, 20)
    tight = cv2.Canny(blurred, 255, 250)
    auto = auto_canny(blurred)

    # show the images
    cv2.imshow('Original', image)
    # cv2.imshow('Edges', np.hstack([wide, tight, auto]))
    cv2.imshow('wide', wide)
    cv2.imshow('tight', tight)
    cv2.imshow('auto', auto)
    cv2.waitKey(0)

def holistically_nested_edge_image(img, image_path = True): 
    '''
    use the hlistically nested edge to detect the edges
    img: file name or image itself (read by the imread)
    image_path = True => img is the path
            = False => already the image, datasets
    '''
    args = {'image': img, 'edge_detector': './holistically-nested-edge-detection/hed_model'}

    roots = './holistically-nested-edge-detection/hed_model'
    protoPath = os.path.join(args['edge_detector'], 'deploy.prototxt')
    modelPath = os.path.join(args['edge_detector'], 'hed_pretrained_bsds.caffemodel')

    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # register our new layer with the model
    cv2.dnn_registerLayer('Crop', CropLayer)

    # load the input image and grab its dimensions
    if image_path: 
        image = cv2.imread(args['image'])
    else: 
        image = args['image']
    (H, W) = image.shape[:2]

    # convert the image to grayscale, blur it, and perform Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # canny = cv2.Canny(blurred, 30, 150)
    canny = cv2.Canny(gray, 20, 30) # to show the paper edge, stop to do the blur

    # construct a blob out of the input image for the Holistically-Nested Edge Detector
    blob = cv2.dnn.blobFromImage(image, scalefactor = 1.0, size = (W, H), mean = (104.00698793, 116.66876762, 122.67891434), swapRB = True, crop = True)

    # set the blob as the input to the network and perform a forward pass to compute the edges
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype('uint8')

    # show the output edge detection results for Canny and Holistically-Nested Edge Detection
    cv2.imshow('Input', image)
    cv2.imshow('Canny', canny)
    cv2.imshow('HED', hed)
    cv2.waitKey(0)
def add_sharpen_kernel(img): 
    '''
    this is to sharpent the image
    img is the image path
    '''
    image = cv2.imread(img)
    cv2.imshow('original', image)

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

def holistically_nested_edge_video(video_path):
    '''
    use the hlistically nested edge to detect the edges
    img: file name or image itself (read by the imread)
    image_path = True => img is the path
            = False => already the image, datasets
    '''
    args = {'input': video_path, 'edge_detector': './holistically-nested-edge-detection/hed_model'}

    roots = './holistically-nested-edge-detection/hed_model'
    protoPath = os.path.join(args['edge_detector'], 'deploy.prototxt')
    modelPath = os.path.join(args['edge_detector'], 'hed_pretrained_bsds.caffemodel')

    # get the video
    # vs = cv2.VideoCapture(args['input'])
    fvs = FileVideoStream(args['input']).start()
    # time.sleep(1.0)

    fps = FPS().start()

    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # register our new layer with the model
    cv2.dnn_registerLayer('Crop', CropLayer)

    # loop over frames from the video streams
    # while True:
    while fvs.more(): 

        # grab the next frame and handle if we are reading from either videocapture or video stream
        # frame = vs.read()[1]
        frame = fvs.read()
        
        # # for the vs part
        # if frame is None: 
        #     break

        frame = imutils.resize(frame, width = 500)
        (H, W) = frame.shape[:2]

        # convert the image to grayscale, blur it, and perform Canny edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # canny = cv2.Canny(blurred, 30, 150)
        canny = cv2.Canny(gray, 30, 150) # to show the paper edge, stop to do the blur

        # construct a blob out of the input image for the Holistically-Nested Edge Detector
        blob = cv2.dnn.blobFromImage(frame, scalefactor = 1.0, size = (W, H), mean = (104.00698793, 116.66876762, 122.67891434), swapRB = False, crop = False)

        # set the blob as the input to the network and perform a forward pass to compute the edges
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0, 0], (W, H))
        hed = (255 * hed).astype('uint8')

        # show the output edge detection results for Canny and Holistically-Nested Edge Detection
        cv2.imshow('Frame', frame)
        cv2.imshow('Canny', canny)
        cv2.imshow('HED', hed)
        key = cv2.waitKey(1) & 0xFF

        fps.update()

        if key == ord('q'): 
            break
    
    fps.stop()
    # vs.release()
    cv2.destroyAllWindows()
    fvs.stop()

if __name__=='__main__': 
    

    file_s_1 = 'test_frame29221.jpg'    
    file_s_2 = 'test_frame29465.jpg'

    file_t_1 = 'test_frame29223.jpg'
    file_t_2 = 'test_frame29467.jpg'

    file_m_1 = 'test_frame29249.jpg' # pass the middle

    file_e_1 = 'test_frame29255.jpg' # most to the end

    file_no_1 = 'test_frame29261.jpg' # no edge 

    root_img = '../vedio_read/data/test_frame'

    test_img = './holistically-nested-edge-detection/images/cat.jpg'

    videopath = '../vedio_read/Gawain-2079_cut.mp4'
    # videopath = '../vedio_read/G2079_cut.avi'
    
    # canny_no_parameter_compare(os.path.join(root_img, file_t_1))
    
    # get the sharpened image and deal with the holistically nested edge method
    # sharpened_image = add_sharpen_kernel(os.path.join(root_img, file_s_1))
    # holistically_nested_edge_image(sharpened_image, image_path= False)

    holistically_nested_edge_video(videopath)


    
    