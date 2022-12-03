"""Funtions for working with signals"""

import math
import numpy as np
import pandas as pd
import scipy.signal
from numpy import matlib


def find_elbow(series):
    """
    Method to find an elbow of a curve.
    References:
        - https://stackoverflow.com/a/2022348
    """
    coords = np.vstack((range(len(series)), series)).T

    # Get the first point
    first_point = coords[0]
    # Get vector between first and last point - this is the line
    line = coords[-1] - coords[0]
    line_norm = line / np.sqrt(np.sum(line**2))

    # Find the distance from each point to the line:
    # Vector between all points and first point
    from_first = coords - first_point

    # To calculate the distance to the line, we split from_first into two
    # components, one that is parallel to the line and one that is perpendicular.
    # Then, we take the norm of the part that is perpendicular to the line and
    # get the distance.
    # We find the vector parallel to the line by projecting from_first onto
    # the line. The perpendicular vector is from_first - from_first_parallel
    # We project from_first by taking the scalar product of the vector with
    # the unit vector that points in the direction of the line (this gives us
    # the length of the projection of from_first onto the line). If we
    # multiply the scalar product by the unit vector, we have from_first_parallel
    prod = np.sum(from_first * np.matlib.repmat(line_norm, len(series), 1), axis=1)
    from_first_parallel = np.outer(prod, line_norm)
    to_line = from_first - from_first_parallel

    # Distance to line is the norm of to_line
    dist = np.sqrt(np.sum(to_line ** 2, axis=1))
    # Knee/elbow is the point with max distance value
    idx = np.argmax(dist)

    return dist, idx

def butter_lowpass(cutoff, freq, order=5):
    """Butterworth lowpass filter"""
    nyq = 0.5 * freq
    normal_cutoff = cutoff / nyq
    numerator, denominator = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return numerator, denominator

def butter_lowpass_filter(sig, cutoff, freq, order=5):
    """Apply filter to signal"""
    numerator, denominator = butter_lowpass(cutoff, freq, order=order)
    return scipy.signal.lfilter(numerator, denominator, sig)

def window_smoothing(sig_y, window_size, window_function):
    """Smooth signal by averging using a specific window function"""
    filtered_y = pd.DataFrame(sig_y).rolling(
        window=window_size,
        win_type=window_function,
        center=True
    ).mean().squeeze().fillna(0.0) # Smoooooth af
    # First 'window_size' values of the filtered signal have to be discarded (are NaNs anyway).
    # Smmothed values corredpond to the center position within the windows.
    # Therefore, to align the filtered and un-filtered signals,
    # the indexing sequence ('sig_x' in this case)
    # has to be sliced by half of 'window_size' at the beginning and end.
    # sliced_x = sig_x[window_size // 2:-window_size // 2]
    return filtered_y.values

def dtw(series1, series2, dist_fn, win=None):
    """
    Calculate dynamic time warping cost matrix
    References:
        - Omer Gold and Micha Sharir.
          Dynamic Time Warping and Geometric Edit Distance: Breaking the Quadratic Barrier.
          ACM Trans. Algorithms 14, 4, Article 50. (2018). DOI:https://doi.org/10.1145/3230734
        - https://github.com/pierre-rouanet/dtw
    """
    # Calculate costs
    if win is not None:
        dist = np.full((len(series1) + 1, len(series2) + 1), np.inf)
        for idx in range(1, len(series1) + 1):
            dist[idx, max(1, idx - win[0]):min(len(series1) + 1, idx + win[1] + 1)] = 0
        dist[0, 0] = 0
    else:
        dist = np.zeros((len(series1) + 1, len(series2) + 1))
        dist[0, 1:] = np.inf
        dist[1:, 0] = np.inf

    acc_dist = dist[1:, 1:]

    for idx, __ in enumerate(series1):
        for jdx, __ in enumerate(series2):
            if win is None or (max(0, idx - win[0]) <= jdx <= min(len(series2), idx + win[1])):
                acc_dist[idx, jdx] = dist_fn(series1[idx] - series2[jdx])

    # Calculate accumulated costs
    jrange = range(len(series2))
    for idx, __ in enumerate(series1):
        if win is not None:
            jrange = range(max(0, idx - win[0]), min(len(series2), idx + win[1] + 1))
        for jdx in jrange:
            acc_dist[idx, jdx] += min(
                [
                    dist[min(idx + 1, len(series1)), jdx],
                    dist[idx, min(jdx + 1, len(series2))],
                    dist[idx, jdx]
                ]
            )

    # Traceback
    idx, jdx = np.array(np.shape(dist)) - 2
    path_x, path_y = [idx], [jdx]
    while (idx > 0) or (jdx > 0):
        t_b = np.argmin((dist[idx, jdx], dist[idx, jdx + 1], dist[idx + 1, jdx]))
        if t_b == 0:
            idx -= 1
            jdx -= 1
        elif t_b == 1:
            idx -= 1
        else:
            jdx -= 1
        path_x.insert(0, idx)
        path_y.insert(0, jdx)

    return acc_dist, [path_x, path_y]

