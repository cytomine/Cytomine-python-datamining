# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy
from shapely.validation import explain_validity
from shapely.geometry import Polygon
from shapely.affinity import affine_transform as aff_transfo
from functools import partial
from scipy.ndimage.morphology import binary_hit_or_miss

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__contributors__ = ["Romain Mormont <r.mormont@student.ulg.ac.be>"]
__version__ = "0.1"


def affine_transform(xx_coef=1, xy_coef=0, yx_coef=0, yy_coef=1,
                     delta_x=0, delta_y=0):
    """
    Represents a 2D affine transformation:

    x' = xx_coef * x + xy_coef * y + delta_x
    y' = yx_coef * x + yy_coef * y + delta_y

    Constructor parameters
    ----------------------
    xx_coef: float (default: 1)
        The x from x coefficient
    xy_coef: float (default: 0)
        The x from y coefficient
    yx_coef: float (default: 0)
        The y from x coefficient
    yy_coef: float (default: 1)
        The y from y coefficient
    delta_x: float (default: 0)
        The translation over x-axis
    delta_y: float (default: 0)
        The translation over y-axis

    Return
    ------
    affine_transformer : callable: shapely.Geometry => shapely.Geometry
        The function representing the 2D affine transformation
    """
    return partial(aff_transfo, matrix=[xx_coef, xy_coef, yx_coef, yy_coef,
                                        delta_x, delta_y])


def identity(x):
    """Identity function

    Parameters
    ----------
    x: T
        The object to return

    Returns
    -------
    x: T
        The passed object
    """
    return x


def process_mask(mask):
    """Remove patterns from the mask that'd yield invalid polygons
    Inspired from: https://goo.gl/fTxuqk
    Parameters
    ----------
    mask: ndarray
        The binary mask

    Returns
    -------
    cleaned: ndarray
        The cleaned mask
    """
    structures = list()
    # remove down-left to up-right diagonal pattern
    structures.append((np.array([[0, 0, 1], [0, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])))
    # remove up-left to down-right diagonal pattern
    structures.append((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])))
    # Does removing the second pattern can recreate the first one ? If so, how to avoid it? (iterative way?)
    # remove up line
    structures.append((np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]]), np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])))
    # remove down line
    structures.append((np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]]), np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]])))
    # remove left line
    structures.append((np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]), np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])))
    # remove right line
    structures.append((np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]])))

    for struct1, struct2 in structures:
        pattern_mask = binary_hit_or_miss(mask, structure1=struct1, structure2=struct2).astype(np.uint8)
        pattern_mask[pattern_mask == 1] = 255
        pattern_mask[pattern_mask == 0] = 0
        mask = mask - pattern_mask

    return mask


class Locator(object):
    """A class providing methods for extracting polygons from a binary mask.
    """
    def locate(self, segmented, offset=None):
        """Extract polygons for the foreground elements of the segmented image.
        Inspired from: https://goo.gl/HYPrR1
        Parameters
        ----------
        segmented: ndarray (shape: (width, height))
            An NumPy representation of a segmented image. Background pixels are represented by
            the value 0 ('black') while foreground ones are represented by the value 255 ('white').
            The type of the array values should be 'uint8'.
        offset: (int, int) (optional, default: (0,0))
            An offset indicating the coordinates of the top-leftmost pixel of the segmented image in the
            original image.
        Returns
        -------
        polygons : iterable (subtype: shapely.geometry.Polygon)
            An iterable containing the polygons extracted from the segmented image. The reference
            point (0,0) for the polygons coordinates is the upper-left corner of the initial image.
        """
        # clean invalid patterns from the mask
        segmented = process_mask(segmented)

        # CV_RETR_EXTERNAL to only get external contours.
        contours, hierarchy = cv2.findContours(segmented.copy(),
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Note: points are represented as (col, row)-tuples apparently
        transform = identity
        if offset is not None:
            col_off, row_off = offset
            transform = affine_transform(delta_x=col_off, delta_y=row_off)
        components = []
        if len(contours) > 0:
            top_index = 0
            tops_remaining = True
            while tops_remaining:
                exterior = contours[top_index][:, 0, :].tolist()

                interiors = []
                # check if there are childs and process if necessary
                if hierarchy[0][top_index][2] != -1:
                    sub_index = hierarchy[0][top_index][2]
                    subs_remaining = True
                    while subs_remaining:
                        interiors.append(contours[sub_index][:, 0, :].tolist())

                        # check if there is another sub contour
                        if hierarchy[0][sub_index][0] != -1:
                            sub_index = hierarchy[0][sub_index][0]
                        else:
                            subs_remaining = False

                # add component tuple to components only if exterior is a polygon
                if len(exterior) > 3:
                    polygon = Polygon(exterior, interiors)
                    polygon = transform(polygon)
                    if polygon.is_valid:  # some polygons might be invalid
                        components.append(polygon)
                    else:
                        print explain_validity(polygon)

                # check if there is another top contour
                if hierarchy[0][top_index][0] != -1:
                    top_index = hierarchy[0][top_index][0]
                else:
                    tops_remaining = False

        del contours
        del hierarchy
        return components
