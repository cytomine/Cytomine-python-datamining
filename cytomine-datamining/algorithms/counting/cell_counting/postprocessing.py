# -*- coding: utf-8 -*-
import numpy as np

from scipy import ndimage as snd

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def non_maximum_suppression(scoremap, post_sigma=None,
                            post_threshold=0.0, post_min_dist=7):
    """
    Extract local maxima whose value is higher a given threshold and that have
    at least at a given distance between them. An optional smoothing can be
    performed before to merge multiple peaks.

    Parameters
    ----------
    scoremap: array-like of shape (width, height)
        The image where to find local maxima.
    post_sigma: float, None, optional (default=None)
        The standard deviation for optional smoothing.
    post_threshold: float, optional (default=0.0)
        The discarding threshold.
    post_min_dist: float, optional (default=7)
        The minimum distance between two peaks.

    Returns
    -------
    mask: array-like of shape (width, height)
        The mask containing local maxima.

    Notes
    -----
    . The code is adapted from skimage library.
    """
    scoremap = scoremap.astype(np.float)

    # Normalize
    scoremap /= np.max(scoremap)

    # Optional Gaussian smoothing
    if post_sigma is not None and post_sigma > 0:
        scoremap = snd.filters.gaussian_filter(scoremap, sigma=post_sigma)

    # Re-normalize after smoothing !
    # scoremap /= np.max(scoremap)

    # A maximum filter is used for finding local maxima. This operation dilates
    # the original image and merges neighboring local maxima closer than the size
    # of the dilation.
    # Locations where the original image is equal to the dilated image are
    # returned as local maxima.

    # Non maximum filter
    size = 2 * post_min_dist + 1
    image_max = snd.maximum_filter(scoremap, size=size, mode='reflect')
    mask = scoremap == image_max

    # Zero out the image borders
    for i in range(mask.ndim):
        mask = mask.swapaxes(0, i)
        remove = (2 * post_min_dist)
        mask[:remove // 2] = mask[-remove // 2:] = False
        mask = mask.swapaxes(0, i)

    # Find top peak candidates
    mask &= scoremap > post_threshold
    return mask
