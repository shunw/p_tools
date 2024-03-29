import matplotlib.pyplot as plt
import numpy as np

def detect_blur_fft(image, size = 60, thresh = 10, vis = False):
    # grab the dimentions of the image and use the dimensions to derive the center
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the freq transform, then shift the zero freq component (i.e., DC component located at the top-left corner) to the center where it will be more easy to analyze
    fft = np.fft.fft2(image)
    
    fftShift = np.fft.fftshift(fft)
    

    # check to see if we are visualizing our output
    if vis: 

        # compute the magnitude spectrum of the transform
        magnitude = 20 * np.log(np.abs)

        # display the original input image
        (fig, ax) = plt.subplot(1, 2, )
        ax[0].imshow(image, cmap = 'gray')
        ax[0].set_title('Input')
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        # display the magnitude image
        ax[1].imshow(magnitude, cmap = 'gray')
        ax[1].set_title('Magnitude Spectrum')
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low freq), apply the inverse shift such that the DC component once again becomes the top-left, and then apply the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)

    recon = np.fft.ifft2(fftShift)


    # compute the magnitude spectrum of the reconstructed image, then compte the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)

    # the image will be considered 'blurry' if the mean value of the magnitude is less than the threshold value
    return (mean, mean <= thresh)