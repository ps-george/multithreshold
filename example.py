import cv2
import otsu
import matplotlib.pyplot as plt
import numpy as np

def main():
    L = 256 # number of levels
    M = 32  # number of bins for bin-grouping normalisation

    N = L // M

    # Load original image
    img = cv2.imread('data/soil.jpg', 0) # read image in as grayscale
    plt.figure()
    plt.imshow(img, cmap='gray')

    # Blur image
    img = cv2.GaussianBlur(img, (7,7), 0)
    img = cv2.GaussianBlur(img, (7,7), 0)
    plt.figure()
    plt.imshow(img, cmap='gray')

    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    # Normalise bin histogram
    norm_hist = otsu.normalised_histogram_binning(hist, M=M, L=L)
    plt.figure()
    plt.bar(range(0, norm_hist.shape[0]), norm_hist.ravel())
    
    # Estimate valley regions
    valleys = otsu.find_valleys(norm_hist)
    print(valleys)
    thresholds = otsu.threshold_valley_regions(hist, valleys, N) 
    print(thresholds)

    otsu_threshold = otsu.otsu_method(hist)
    thresholds = otsu.modified_TSMO(hist, M=M, L=L)
    print(otsu_threshold, thresholds)

    # Show histogram with thresholds
    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())
    plt.axvline(x=otsu_threshold, color='k')
    for t in thresholds:
        plt.axvline(x=t, color='red')

    plt.show()

    
if __name__=='__main__':
    main()
