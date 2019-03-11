import cv2
import otsu
import matplotlib.pyplot as plt
import numpy as np


def show_thresholds(src_img, dst_img, thresholds):
    """Visualise thresholds."""
    colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 128, 0), (0, 204, 102),
              (51, 255, 255), (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255), (255, 0, 127)]
    masks = otsu.multithreshold(src_img, thresholds)
    for i, mask in enumerate(masks):
        # for debugging masks
        # background = np.zeros((dst_img.shape[0], dst_img.shape[1], 3))
        # background[mask] = (255, 255, 255)
        # plt.figure()
        # plt.imshow(background)
        dst_img[mask] = colors[i]
    return dst_img


def full_method(img, L=256, M=32):
    """Obtain thresholds and image masks directly from image.

    Abstracts away the need to handle histogram generation.
    """
    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    thresholds = otsu.modified_TSMO(hist, M=M, L=L)
    masks = otsu.multithreshold(img, thresholds)
    return thresholds, masks


def main(path='images/soil.jpg'):
    L = 256  # number of levels
    M = 32  # number of bins for bin-grouping normalisation

    N = L // M
    # Load original image
    img = cv2.imread(path, 0)  # read image in as grayscale

    # Blur image to denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate histogram
    hist = cv2.calcHist(
        [img],
        channels=[0],
        mask=None,
        histSize=[L],
        ranges=[0, L]
    )

    ### Do modified TSMO step by step
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
    ###

    # modified_TSMO does all the steps above
    thresholds2 = otsu.modified_TSMO(hist, M=M, L=L)

    # Threshold obtained through default otsu method.
    otsu_threshold, _ = otsu.otsu_method(hist)

    print('Otsu threshold: {}\nStep-by-step MTSMO: {}\nMTSMO: {}'.format(
        otsu_threshold, thresholds, thresholds2))

    # Show histogram with thresholds
    plt.figure()
    plt.bar(range(0, hist.shape[0]), hist.ravel())
    plt.axvline(x=otsu_threshold, color='k')
    for t in thresholds:
        plt.axvline(x=t, color='red')

    # Illustrate thresholds
    img_1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_3 = img_1.copy()
    img_auto = img_1.copy()

    show_thresholds(img, img_1, [thresholds[0]])
    show_thresholds(img, img_3, thresholds[0:3])
    show_thresholds(img, img_auto, thresholds)

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
    ax.set_title('1 level')
    plt.imshow(img_1)
    plt.show()

if __name__ == '__main__':
    main()
