import numpy as np


def otsu_method(hist):
    """
    Optimized implementation of Otsu's Method algorithm.

    Adapted from https://github.com/subokita/Sandbox/blob/master/otsu.py and the Matlab implementation on Wikipedia.
    """
    num_bins = hist.shape[0]
    total = hist.sum()
    sum_total = np.dot(range(0, num_bins), hist)

    weight_background = 0.0
    sum_background = 0.0

    optimum_value = 0
    maximum = -np.inf

    for t in range(0, num_bins):
        # background weight will be incremented, while foreground's will be reduced
        weight_background += hist.item(t)
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * hist.item(t)
        mean_foreground = (sum_total - sum_background) / weight_foreground
        mean_background = sum_background / weight_background

        inter_class_variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground) ** 2

        # find the threshold with maximum variances between classes
        if inter_class_variance > maximum:
            optimum_value = t
            maximum = inter_class_variance
    return optimum_value, maximum


def otsu(hist):
    pos, val = otsu_method
    return pos


def normalised_histogram_binning(hist, M=32, L=256):
    """Normalised histogram binning"""
    norm_hist = np.zeros((M, 1), dtype=np.float32)
    N = L // M
    counters = [range(x, x+N) for x in range(0, L, N)]
    for i, C in enumerate(counters):
        norm_hist[i] = 0
        for j in C:
            norm_hist[i] += hist[j]
    norm_hist = (norm_hist / norm_hist.max()) * 100
    return norm_hist


def find_valleys(H):
    """Valley estimation on *H*, H should be normalised-binned-grouped histogram."""
    hsize = H.shape[0]
    probs = np.zeros((hsize, 1), dtype=int)
    costs = np.zeros((hsize, 1), dtype=float)
    for i in range(1, hsize-1):
        if H[i] > H[i-1] or H[i] > H[i+1]:
            probs[i] = 0
        elif H[i] < H[i-1] and H[i] == H[i+1]:
            probs[i] = 1
            costs[i] = H[i-1] - H[i]
        elif H[i] == H[i-1] and H[i] < H[i+1]:
            probs[i] = 3
            costs[i] = H[i+1] - H[i]
        elif H[i] < H[i-1] and H[i] < H[i+1]:
            probs[i] = 4
            costs[i] = (H[i-1] + H[i+1]) - 2*H[i]
        elif H[i] == H[i-1] and H[i] == H[i+1]:
            probs[i] = probs[i-1]
            costs[i] = probs[i-1]
    for i in range(1, hsize-1):
        if probs[i] != 0:
            # floor devision. if > 4 = 1, else 0
            probs[i] = (probs[i-1] + probs[i] + probs[i+1]) // 4
    valleys = [i for i, x in enumerate(probs) if x > 0]
    # if maximum is not None and maximum < len(valleys):
    # valleys = sorted(valleys, key=lambda x: costs[x])[0:maximum]
    return valleys


def valley_estimation(hist, M=32, L=256):
    """Valley estimation for histogram. L should be divisible by M."""
    # Normalised histogram binning
    norm_hist = normalised_histogram_binning(hist, M, L)
    valleys = find_valleys(norm_hist)
    return valleys


def threshold_valley_regions(hist, valleys, N):
    """Perform Otsu's method over estimated valley regions."""
    thresholds = []
    for valley in valleys:
        start_pos = (valley * N) - N
        end_pos = (valley + 2) * N
        h = hist[start_pos:end_pos]
        sub_threshold, val = otsu_method(h)
        thresholds.append((start_pos + sub_threshold, val))
    thresholds.sort(key=lambda x: x[1], reverse=True)
    thresholds, values = [list(t) for t in zip(*thresholds)]
    return thresholds


def modified_TSMO(hist, M=32, L=256):
    """Modified Two-Stage Multithreshold Otsu Method.

    Implemented based on description in:
    Huang, D. Y., Lin, T. W., & Hu, W. C. (2011).
    Automatic multilevel thresholding based on two-stage Otsu’s method with cluster determination by valley estimation.
    International Journal of Innovative Computing, Information and Control, 7(10), 56315644.
    """

    N = L // M
    valleys = valley_estimation(hist, M, L)
    thresholds = threshold_valley_regions(hist, valleys, N)
    return thresholds
