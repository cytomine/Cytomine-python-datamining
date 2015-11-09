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
try:
    import Image
except ImportError:
    from PIL import Image
import numpy as np
from ..indexer import Sliceable
from ..datatype import PILConverter


class ImageBuffer(Sliceable):
    """
    ===========
    ImageBuffer
    ===========
    An :class:`ImageBuffer` is a buffer of predefined length which returns
    images. Each seed refers to a new image.

    The [int] operator returns an image
    The [int:int] operator returns an :class:`ImageBuffer`

    Image class
    -----------
    By default, the images are :class:`PIL.Image`. Another format (Numpy,
    openCV, etc.) can be chosen by providing the appropriate converter (see
    :class:`ImageConverter`) to the image loader

    Constructor parameters
    ----------------------
    seed_sequence : iterable of objects
        A seed contains all the information required to load the image it spans
    image_loader : :class:`ImageLoader`
        An :class:`ImageLoader` instance which can work with the seeds given
    """

    def __init__(self, seed_sequence, image_loader):
        self.seeds = seed_sequence
        self.loader = image_loader

    def _get(self, index):
        return self.loader.load(self.seeds[index])

    def _slice(self, shallow_copy, slice_range):
        shallow_copy.seeds = self.seeds[slice_range]

    def __len__(self):
        return len(self.seeds)


class ImageLoader:
    """
    ===========
    ImageLoader
    ===========
    A class which can load images

    Constructor parameters
    ----------------------
    image_converter : ImageConverter
        A converter to get the appropriate format
    """

    __metaclass__ = ABCMeta

    def __init__(self, image_converter=PILConverter()):
        self.converter = image_converter

    @abstractmethod
    def _load(self, seed):
        """
        Load an image based on the seed. The tile wrapper is constructed
        in the :method:`load` method.

        Parameters
        ----------
        seed : ImageLoader dependant (default : None)
            A seed to load the image

        Return
        ------
        image : ImageLoader dependant
            The corresponding image
        """
        pass

    def load(self, seed=None):
        """
        Load a tile based on the seed. The tile has a col_offset and row_offset
        of 0. It must be reimplimented if this is not the expected behavior.

        Parameters
        ----------
        seed : ImageLoader dependant (default : None)
            A seed to load the image

        Return
        ------
        image : ImageLoader dependant
            The corresponding image
        """
        img = self._load(seed)
        return self.converter.convert(img)


class PILLoader(ImageLoader):
    """
    =========
    PILLoader
    =========
    A class which can load image files into :class:`PIL.Image`.
    See the PIL library for more information.
    """
    def __init__(self, image_converter=PILConverter()):
        ImageLoader.__init__(self, image_converter)

    def _load(self, image_file):
        """
        Load a tile from a file

        Parameters
        ----------
        image_file : str or file
            The path to the file

        Return
        ------
        tile : PIL.Image
            The corresponding image
        """
        return Image.open(image_file)


class NotCMapPILLoader(PILLoader):
    """
    ==================
    NotCMapImageLoader
    ==================
    Load image file and convert palette into RGB if necessary
    """
    def __init__(self, image_converter=PILConverter()):
        PILLoader.__init__(self, image_converter)

    def _load(self, image_file):
        image = Image.open(image_file)

        if image.mode == "P":
            image = image.convert("RGB")
        return image


class NumpyLoader(ImageLoader):
    """
    ===========
    NumpyLoader
    ===========
    Load a numpy file representing an image
    """
    def __init__(self, image_converter=PILConverter()):
        ImageLoader.__init__(self, image_converter)

    def _load(self, numpy_file):
        """
        Load a image from a file

        Parameters
        ----------
        numpy_file : str or file
            The path to the file

        Return
        ------
        image : numpy array representing an image
            The corresponding image
        """
        return np.load(numpy_file)
