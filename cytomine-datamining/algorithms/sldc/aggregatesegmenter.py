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

import numpy as np
import cv2

import cv
import copy
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.measurements import label
from segmenter import BinarySegmenter
from utilities.source.imageconverter import NumpyConverter


class AggregateSegmenter(BinarySegmenter):
    """
    ==================
    AggregateSegmenter
    ==================
    A :class:`Segmenter` for segmenting cells in aggregate and architectural
    pattern
    """

    def __init__(self, color_deconvoluter, struct_elem,
                 cell_max_area,
                 cell_min_circularity=.8,
                 border=7,
                 ):
        self._color_deconvoluter = color_deconvoluter
        self._struct_elem = struct_elem
        self._cell_max_area = cell_max_area
        self._min_circularity = cell_min_circularity
        self._border = border

    def _otsu_threshold_with_mask(image, mask, mode):

        mask_indices = np.nonzero(mask)

        temp = np.array([image[mask_indices]])

        temp = temp[temp < 120]

        otsu_threshold,_ = cv2.threshold( temp, 128, 255, cv2.THRESH_OTSU | mode)

        _, image = cv2.threshold(image, otsu_threshold, 255, cv2.THRESH_BINARY_INV)

        return otsu_threshold, image

    def segment(self, np_image):
        """
        Parameters
        ----------
        np_image : numpy.ndarray
            A RGBA image to segment
        """
        border = self._border
        alpha = np.array(np_image[:, :, 3])
        image = np.array(np_image[:, :, 0:3])
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_dec = self._color_deconvoluter.transform(temp)
        temp = cv2.cvtColor(im_dec, cv2.COLOR_RGB2GRAY)
        otsu_threshold, internal_binary = self._otsu_threshold_with_mask(temp, alpha, cv2.THRESH_BINARY_INV)
        internal_binary_copy = copy.copy(internal_binary)
        contours2, hierarchy = cv2.findContours(internal_binary_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #filling inclusions without filling inter-cell space
        mask = np.zeros(internal_binary.shape)

        for i, contour in enumerate(contours2):

            if (hierarchy[0, i, 3] != -1): #internal contour

                convex_hull = cv2.convexHull(contour)

                convex_area = cv2.contourArea(convex_hull)

                perimeter = cv2.arcLength(convex_hull,True)

                circularity = 4*np.pi*convex_area / (perimeter*perimeter)

                #visualisation
                #cv2.drawContours(np_image, contour, -1, (0,255,255), -1)

                if (convex_area < self._cell_max_area/10):

                    #removing small objects

                    cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)

                if (convex_area < self._cell_max_area/2) and (circularity > self._min_cirulatiry):

                    #removing potential inclusions

                    cv2.drawContours(internal_binary, [convex_hull], -1, 255, -1)

        internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_CLOSE, self._struct_elem, iterations = 1)

        internal_binary = cv2.morphologyEx(internal_binary,cv2.MORPH_OPEN, self._struct_elem, iterations = 2)

        dt = cv2.distanceTransform(internal_binary, cv2.cv.CV_DIST_L2, 3)

        dt = dt[0]

        im_dec[internal_binary == 0] = (255,255,255)

        #detection maxima locaux
        local_min_ind = maximum_filter(dt, (9,9) ) == dt

        #image markers
        markers = np.zeros(dt.shape).astype(np.uint8)

        #maxima locaux
        markers[local_min_ind] = 255

        #impose tous les markers sont a l'interieur du contour
        markers[internal_binary == 0] = 0

        markers = cv2.dilate(markers,self._struct_elem, iterations = 2)

        markers = markers.astype(np.int32)

        #labellise les composantes connexes 1...nbmarkers
        markers, nb_labels = label(markers, np.ones((3,3)))

        borders = cv2.dilate(internal_binary, self._struct_elem, iterations=1)

        markers[borders == 0] = 0

        borders = borders - internal_binary

        #cadre blanc autour (pour eviter contour de la taille iamge?)
        markers[borders == 255] = nb_labels+2

        markers = cv2.copyMakeBorder(markers, border, border, border, border, cv2.BORDER_CONSTANT, value = nb_labels+2)

        im_dec = cv2.copyMakeBorder(im_dec, border, border, border, border, cv2.BORDER_CONSTANT, value = (255,255,255))

        cv2.watershed(im_dec, markers)

        #enleve cadre
        markers = markers[border:-border, border:-border]

        #repasse en opencv (pour compatibilité objectfinder)
        internal_binary = np.zeros(internal_binary.shape).astype(np.uint8)

        cv_image = cv.CreateImageHeader((internal_binary.shape[1], internal_binary.shape[0]), cv.IPL_DEPTH_8U, 1)
        cv.SetData(cv_image, mask.tostring())
        return NumpyConverter().convert((cv_image))





