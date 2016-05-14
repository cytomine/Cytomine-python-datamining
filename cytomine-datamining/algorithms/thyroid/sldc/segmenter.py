# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


class Segmenter(object):
    """Interface to be implemented by any class which implements a segmentation algorithm
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def segment(self, image):
        """Segment the image using a custom segmentation algorithm
        Parameters
        ----------
        image: ndarray (shape: [width, height{, channels}])
            An NumPy representation of the image to segment.

        Returns
        -------
        segmented : ndarray (shape: [width, height])
            An NumPy representation of the segmented image. Background pixels are represented by
            the value 0 ('black') while foreground ones are represented by the value 255 ('white').
            The type of the array values is 'uint8'.
        """
        pass
