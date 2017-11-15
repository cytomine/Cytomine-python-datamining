# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as snd

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def scoremap_transform_edt(binary_mask, mean_radius, alpha):
    """
    Compute an exponentially shaped Euclidean distance transform 
    of a binary mask.

    Parameters
    ----------
    binary_mask : array-like
        The binary mask.
    mean_radius : integer
        The mean radius of objects contained in the image.
    alpha : integer, optional (default: 4)
        The parameter controlling the shape of the exponential.

    Returns
    -------
    scoremap: array-like
        Transformed binary mask using exponential Euclidean distance
        transform, with each score belonging to [0, 1] interval.

    References
    ----------
    . [1] Sironal et al., "Multiscale Centerline Detection by Learning
          a Scale-Scpace Distance Transform", CVPR, 2014.
    . [2] P. Kainz et al., "You should use regression to detect cells",
          MICCAI, 2015.
    """

    # Reverse mask to have (i,j)=0 if there is an annotation in (i,j)
    binary_mask = 1 - binary_mask

    scoremap = np.asarray(snd.distance_transform_edt(binary_mask))
    st_d_m = np.nonzero(scoremap < mean_radius)
    gt_d_m = np.nonzero(scoremap >= mean_radius)

    scoremap[st_d_m] = np.exp(alpha * (1 - (scoremap[st_d_m] / mean_radius))) - 1
    scoremap[gt_d_m] = 0

    # Normalization to have scores in [0, 1]
    scoremap /= scoremap.max()

    return scoremap


def scoremap_transform_density(binary_mask, mean_radius, use_radius=False, k_factor=1):
    """
    Compute a density map from a binary_mask, using a Gaussian filter.

    Parameters
    ----------
    binary_mask: array-like
        The binary mask.
    mean_radius: integer
        The mean radius of objects contained in the image.
    use_radius: boolean (default: False)
        If radius has to be used. If not, sigma=1
    k_factor: integer (default: 1)
        By how many scale the gaussian filtered scoremap

    Returns
    -------
    scoremap: array-like
        Transformed binary mask using a Gaussian filter, where each 
        element (i,j) is the density of object per pixel at position
        (i,j) with values in [0, 1] interval.

    References
    ----------
    . [1] L. Fiaschi et al., "Learning to count with regression forest 
          and structured labels", ICPR, 2012.
    """
    if not use_radius:
        mean_radius = 2
    sigma = mean_radius / 2.

    binary_mask = binary_mask.astype(np.float)
    scoremap = float(k_factor) * snd.filters.gaussian_filter(binary_mask, sigma=sigma)

    return scoremap
