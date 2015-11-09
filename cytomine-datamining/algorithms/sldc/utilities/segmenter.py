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

from abc import ABCMeta, abstractmethod
import numpy as np
import cv2

class Segmenter:
    """
    =========
    Segmenter
    =========
    An interface for :class:`Segmenter` whose role is to segment the given
    image according to the instance policy.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def segment(self, np_image):
        """
        Segment the given image (not in place if not explicitely speficied).

        Note
        ----
        See the concrete class description for a description of the appropriate
        formats of numpy.ndarray

        Parameters
        ----------
        np_image : numpy.ndarray
            A numpy array represention of an image

        Return
        ------
        segmented : numpy.ndarray
            A numpy array represention of the segmented image
        """
        pass


class BinarySegmenter(Segmenter):
    """
    ===============
    BinarySegmenter
    ===============
    A :class:`BinarySegmenter`is a :class:`Segmenter` whose outputs images
    with values in the binary set {0, 255}
    """

    __metaclass__ = ABCMeta


class CDSegmenter(BinarySegmenter):
    """
    ===========
    CDSegmenter
    ===========
    A segmenter performing :
    - Color deconvolution (see :class:`ColorDeconvoluter`)
    - Static thresholding (identify cells)
    - Morphological closure (remove holes in cells)
    - Morphological opening (remove small objects)
    - Morphological closure (merge close objects)

    Format
    ------
    the given numpy.ndarrays are supposed to be RGB images :
    np_image.shape = (height, width, color=3) with values in the range
    [0, 255]

    Constructor parameters
    ----------------------
    color_deconvoluter : :class:`ColorDeconvoluter` instance
        The color deconvoluter performing the deconvolution
    threshold : int [0, 255] (default : 120)
        The threshold value. The higher, the more true/false positive
    struct_elem : binary numpy.ndarray (default : None)
        The structural element used for the morphological operations. If
        None, a default will be supplied
    nb_morph_iter : sequence of int (default [1,3,7])
        The number of iterations of each morphological operation.
            nb_morph_iter[0] : number of first closures
            nb_morph_iter[1] : number of openings
            nb_morph_iter[2] : number of second closures
    """

    def __init__(self, color_deconvoluter, threshold=120,
                 struct_elem=None, nb_morph_iter=[1, 3, 7]):
        self._color_deconvoluter = color_deconvoluter
        self._threshold = threshold
        self._struct_elem = struct_elem
        if self._struct_elem is None:
            self._struct_elem = np.array([[0, 0, 1, 1, 1, 0, 0],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1],
                                         [0, 1, 1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 1, 0, 0], ],
                                        dtype=np.uint8)
        self._nb_morph_iter = nb_morph_iter

    def segment(self, np_image):
        tmp_image = self._color_deconvoluter.transform(np_image)

        #Static thresholding on the gray image
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(tmp_image, self._threshold, 255, cv2.THRESH_BINARY_INV)

        #Remove holes in cells in the binary image
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._struct_elem, iterations=self._nb_morph_iter[0])

        #Remove small objects in the binary image
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._struct_elem, iterations=self._nb_morph_iter[1])

        #Union architectural paterns
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._struct_elem, iterations=self._nb_morph_iter[2])

        return binary
