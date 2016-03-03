# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class Segmenter(object):
    """
    Interface to be implemented by any segmenter
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def segment(self, image):
        """Segment the image
        Parameters
        ----------
        image: array-like, shape = [width, height{, channels}]
            An array-like representation of the image to segment.
        Returns
        -------
        segmented : array-like, shape = [width, height]
            An array-like representation of the segmented image. Background pixels are represented by
            the value 0 ('black') while foreground ones are represented by the value 255 ('white').
            The type of the array values is 'uint8'.
        """
        pass
