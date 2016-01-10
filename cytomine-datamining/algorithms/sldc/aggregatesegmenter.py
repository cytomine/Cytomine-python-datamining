# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'

import cv2
import copy
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from helpers.datamining.segmenter import BinarySegmenter
from helpers.datamining.segmenter import otsu_threshold_with_mask

class AggregateSegmenter(BinarySegmenter):
    """
    ==================
    AggregateSegmenter
    ==================
    A :class:`Segmenter` for segmenting cells in aggregate and architectural
    pattern
    """
    def __init__(self, color_deconvoluter, struct_elem, cell_max_area=4000, cell_min_circularity=.8):
        self._color_deconvoluter = color_deconvoluter
        self._struct_elem = struct_elem
        self._cell_max_area = cell_max_area
        self._min_circularity = cell_min_circularity
        self._small_struct_element = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]
        ]).astype(np.uint8)

    def segment(self, np_image):
        """
        Parameters
        ----------
        np_image : numpy.ndarray
            A RGBA image to segment
        """
        # Extract alpha mask and RGB, cast to uint8 to match opencv types
        alpha = np.array(np_image[:, :, 3]).astype(np.uint8)
        image = np.array(np_image[:, :, 0:3]).astype(np.uint8)

        # Perform color conversions and deconvolution
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_dec = self._color_deconvoluter.transform(image_rgb)
        image_grey = cv2.cvtColor(image_dec, cv2.COLOR_RGB2GRAY)

        # Find and apply Otsu threshold
        otsu_threshold, internal_binary = otsu_threshold_with_mask(image_grey, alpha, cv2.THRESH_BINARY_INV)
        internal_binary_copy = copy.copy(internal_binary)

        # Find interior contours (possibly inclusion that were missed because of their color) and remove the
        # corresponding artifacts
        contours2, hierarchy = cv2.findContours(internal_binary_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Remove contours that don't have a parent contour (???)
        #   contours2 is somtimes so big that calling enumerate on it causes freezes
        #   this pre-filtering to prevent from enumerating this big object
        contour_idx = [i for i, h in enumerate(hierarchy[0]) if h[3] >= 0]

        mask = np.zeros(internal_binary.shape)

        for i in contour_idx:
            contour = contours2[i] # fetch the contour to evaluate

            # Filter contour to make sure it should be filled, get convex hull of the contour to avoid artifact
            convex_hull = cv2.convexHull(contour)
            convex_area = cv2.contourArea(convex_hull)
            perimeter = cv2.arcLength(convex_hull,True)
            circularity = 4 * np.pi * convex_area / (perimeter * perimeter)

            # Remove small objects (first cleaning)
            if convex_area < (self._cell_max_area / 10):
                cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)

            # Remove potential inclusions
            if (convex_area < self._cell_max_area/2) and (circularity > self._min_circularity):
                cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)

        # Deletion of small holes
        internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_CLOSE, self._struct_elem, iterations = 1)
        internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_OPEN, self._struct_elem, iterations = 2)

        # Watershed in order to separate neighbouring arrays
        #  1) Find the markers for starting the watershed (using distance transform)
        #    -> find local maximum of the distance transform
        #    -> create a mask for the markers : (255) for the markers, and (0) for the rest
        #  2)
        # TODO test dt on dilated internal_binary
        dt = cv2.distanceTransform(internal_binary, cv2.cv.CV_DIST_L2, 3)

        #image_dec[internal_binary == 0] = [255,255,255] # (????)

        # Create borders to be added to the markers' image
        borders = cv2.dilate(internal_binary, self._struct_elem, iterations=1)
        borders = borders - cv2.erode(borders, None)

        # Detection maxima locaux
        local_max_ind = maximum_filter(dt, size=9) == dt

        # Create marker mask
        markers = np.zeros(dt.shape).astype(np.uint8)
        markers[local_max_ind] = 255
        markers[internal_binary == 0] = 0

        # Dilate marker to make them more homogeneous
        markers = cv2.dilate(markers, self._small_struct_element, iterations = 2)
        markers = markers.astype(np.int32)

        # Differentiates the colors of the markers (labelling)
        markers, nb_labels = label(markers, np.ones((3,3)))
        # Use 255 as color for borders to distinguish pixels from the background
        if nb_labels < 255:
            markers[markers == 255] = 256
            markers[borders == 255] = 255
        else:
            markers[markers == 255] = nb_labels + 1
            markers[borders == 255] = 255

        # Execute watershed
        cv2.watershed(image_dec, markers)

        # Remove barriers produced by watershed
        markers[markers == -1] = 0
        markers[markers == 255] = 0
        markers[markers > 0] = 255
        return markers.astype(np.uint8)
