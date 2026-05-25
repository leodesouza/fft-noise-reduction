# https://docs.opencv.org/4.13.0/de/dbc/tutorial_py_fourier_transform.html
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('img/messi.jpg',cv.IMREAD_REDUCED_GRAYSCALE_2)
assert img is not None, "file can not be read"
    
# get the frequency transform
f = np.fft.fft2(img)

#now once you got the result, zero frequency component (DC component) will be at top left corner. 
# If you want to bring it to center, you need to shift the result by in both the direction
fshift = np.fft.fftshift(f)

def fft_numpy():    
    #Finding the magnitude spectrum from frequency transform
    magnitude_spectrum = 20*np.log(np.abs(fshift))    

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.title("Input")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum")
    plt.xticks([])
    plt.yticks([])
    
    plt.show()

def apply_high_pass_filter():    
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    
    fshift_filtered = fshift.copy()
    #removing low frequencies
    fshift_filtered[crow-30:crow+31, ccol-30:ccol+31] = 0

    f_shift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_shift)
    img_back = np.real(img_back)

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title("Input")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    plt.imshow(img_back, cmap='jet')
    plt.title('Image after HPF') # High Pass Filter 
    plt.xticks([])
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(img_back)
    plt.title("Result in JET")
    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    #fft_numpy()
    apply_high_pass_filter()