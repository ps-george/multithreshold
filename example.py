import cv2
import otsu
import matplotlib.pyplot as plt
import numpy as np


def main():
    L = 256  # number of levels
    M = 32  # number of bins for bin-grouping normalisation

    N = L // M
    # Load original image
    img = cv2.imread('data/soil.jpg', 0)  # read image in as grayscale
    # plt.figure()
    # plt.imshow(img, cmap='gray')

    # Blur image
    img = cv2.GaussianBlur(img, (7, 7), 0)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    # plt.figure()
    # plt.imshow(img, cmap='gray')

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

    # Estimate valley regions
    valleys = otsu.find_valleys(norm_hist)
    # plot valleys
    plt.figure()
    plt.bar(range(0, norm_hist.shape[0]), norm_hist.ravel())
    for v in valleys:
        plt.axvline(x=v, color='red')

    thresholds = otsu.threshold_valley_regions(hist, valleys, N)

    otsu_threshold, _ = otsu.otsu_method(hist)
    thresholds2 = otsu.modified_TSMO(hist, M=M, L=L, maximum=4)
    print('Otsu threshold: {}\nStep-by-step MTSMO: {}\nMTSMO: {}'.format(
        otsu_threshold, thresholds, thresholds2))

    # Show histogram with thresholds
    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())
    plt.axvline(x=otsu_threshold, color='k')
    for t in thresholds:
        plt.axvline(x=t, color='red')

    # Illustrate thresholds
    img_3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_auto = img_3.copy()
    img_5 = img_3.copy()

    def show_thresholds(src_img, dst_img, thresholds):
        colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 128, 0), (0, 255, 128),
                  (0, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 127)]
        for i, t in enumerate(sorted(thresholds)):
            retval, mask = cv2.threshold(src_img, t, 255, cv2.THRESH_BINARY)
            mask_ind = (mask > 0)
            dst_img[mask_ind] = colors[i]
        return dst_img

    show_thresholds(img, img_auto, thresholds)
    show_thresholds(img, img_3, thresholds[0:3])
    show_thresholds(img, img_5, thresholds[0:5])

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Original image')
    plt.imshow(img, cmap='gray')
    ax = plt.subplot(2, 2, 2)
    ax.set_title('{} levels (Automatic)'.format(len(thresholds)))
    plt.imshow(img_auto)
    ax = plt.subplot(2, 2, 3)
    ax.set_title('3 levels')
    plt.imshow(img_3)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('5 levels')
    plt.imshow(img_5)
    plt.show()


if __name__ == '__main__':
    main()
