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
from PIL import Image
from abc import ABCMeta, abstractmethod


def to_numpy(image):
    if isinstance(image, np.ndarray):
        return image
    return np.asarray(image)


class OverflowManager(object):
    """
    ===============
    OverflowManager
    ===============
    An :class:`OverflowManager` is responsible for managing overflow value
    from the range [0,255]. This base class does nothing (and therefore leaves
    the default mechanism unchanged, probably letting values wrap around).
    """

    def manage(self, image):
        """
        Enforces the overflow policy

        Parameters
        ----------
        image : numpy.ndarray
            The image to process

        Return
        ------
        processed_image : numpy.ndarray
            The processed image
        """
        return image

    def __call__(self, image):
        """
        Enforces the overflow policy

        Parameters
        ----------
        image : numpy.ndarray
            The image to process

        Return
        ------
        processed_image : numpy.ndarray
            The processed image
        """
        return self.manage(image)


class ClipOverflow(OverflowManager):
    """
    =============
    ClipOverflow
    =============
    This class thresholds the exceeding values. That is, value below 0 are
    forced to 0 and values greater than 255 are pushed down to 255.
    """

    def manage(self, image):
        return image.clip(0, 255)


class HistogramEqualizer(OverflowManager):
    """
    ==================
    HistogramEqualizer
    ==================
    This class performs an histogram equalization so as to ensures the correct
    [0,255] range for every color channel
    """

    def __init__(self):
        raise NotImplementedError("This class is not yet implemented")

    def manage(self, image):
        raise NotImplementedError("This class is not yet implemented")


class ImageConverter(object):
    """
    ==============
    ImageConverter
    ==============
    An :class:`ImageConverter` converts images to another internal image
    representation.

    If you need a default exchange format, numpy is probably a good choice.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def convert(self, image):
        """
        Converts, if need be, the given images into another representation
        specified by the class policy

        Parameters
        ----------
        image :
            a supported image representation
        """
        pass


class NumpyConverter(ImageConverter):
    """
    ==============
    NumpyConverter
    ==============
    An :class:`ImageConverter` which converts images to numpy 2D arrays.
    The supported types are PIL images and openCV images.

    Constructor parameters
    ----------------------
    None.
    """

    def convert(self, image):
        return to_numpy(image)


class PILConverter(ImageConverter):
    """
    ============
    PILConverter
    ============
    An :class:`ImageConverter` which converts images to PIL images.
    The supported types are numpy arrays and openCV images.

    Note : PIL images works on the range [0, 255] and not with real values.
    A :class:`OverflowManager`  is necessary to enforce a policy for the
    overflowing values. The default policy is to clip exceeding values.

    Constructor parameters
    ----------------------
    overflow_manager : OverflowManager (default : :class:`ClipOverflow`)
        the management policy for the overflow
    """
    def __init__(self, overflow_manager=ClipOverflow()):
        self.overflow_manager = overflow_manager

    def convert(self, image):
        if isinstance(image, Image.Image):
            return image
        np_image = to_numpy(image)
        np_corrected = self.overflow_manager.manage(np_image)
        return Image.fromarray(np.uint8(np_corrected))


class CVConverter(ImageConverter):
    """
    ============
    PILConverter
    ============
    An :class:`ImageConverter` which converts images to openCV images.
    The supported types are numpy arrays and PIL images.

    Class constants
    ---------------
    CV : int
        a constant representing the cv version
    CV2 : int
        a constant representing the cv2 version

    Constructor parameters
    ----------------------
    version : int in {CV, CV2} (default : CV)
        The version of openCV array to convert to
    """

    CV = 1
    CV2 = 2

    def __init__(self, version=CV):
        self.version = version

    def convert(self, image):
        if self.version == CVConverter.CV2:
            return to_numpy(image)
        if isinstance(image, cv.cvmat):
            return image
        return cv.fromarray(to_numpy(image))
