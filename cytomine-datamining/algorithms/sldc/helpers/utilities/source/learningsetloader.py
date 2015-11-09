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
import os
from ..indexer import Sliceable
from .imagebuffer import ImageBuffer, NotCMapPILLoader


class LearningSetBuffer(Sliceable):
    """
    =================
    LearningSetBuffer
    =================
    A :class:`LearningSetBuffer` is a buffer of predefined length which returns
    pairs of (image, label).

    The [int] operator returns a pair (image, label)
    The [int:int] operator returns an :class:`LearningSetBuffer`
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get(self, index):
        pass

    @abstractmethod
    def _slice(self, shallow_copy, slice_range):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def unzip(self):
        """
        Dissociate the images and the label. 

        Return
        ------
        (img_buffer, labels)
            img_buffer : :class:`ImageBuffer`
                An :class:`ImageBuffer` view of the images in the learning set
            labels : list of int
                The labels associated to the learning set
        """
        pass


class LearningSetFromDir(LearningSetBuffer):
    """
    ==================
    LearningSetFromDir
    ==================
    A :class:`LearningSetFromDir` loads learning set pairs of (image, label)
    on demand.

    The [int] operator returns a pair (image, label)
    The [int:int] operator returns an :class:`LearningSetFromDir`

    Image class
    -----------
    By default, the images are :class:`PIL.Image`. Another format (Numpy,
    openCV, etc.) can be chosen by providing the appropriate converter (see
    :class:`ImageConverter`) to the image loader

    Directory layout
    ---------------
    The constructor is expecting the path to a directory where each subdirectory
    contains images of the same label. Either the subdirectory names are
    integers and will serve as the label for the images or a mapping dictionary
    must be provided

    Constructor parameters
    ----------------------
    directory : path to a directory
        The directory where the images are stored, grouped by label (see 
        the 'Directory layout' note)
    image_loader : :class:`ImageLoader`
        An :class:`ImageLoader` instance which can work with the seeds given 
    map_classes : dictionary str->int (default : None)
        An optional mapping dictionary which translates subdirectory names
        to label
    """

    def __init__(self, directory, image_loader=NotCMapPILLoader(), 
                 map_classes=None):
        self.loader = image_loader

        # Getting the information from the files
        self.image_paths = []
        self.labels = []

        for c in os.listdir(directory):

            label_path = os.path.join(directory, c)
            for _file in os.listdir(label_path):
                img_path = os.path.join(label_path, _file)
                self.image_paths.append(img_path)
                if map_classes:
                    self.labels.append(map_classes[c])
                else:
                    self.labels.append(int(c))


    def _get(self, index):
        return self.loader.load(self.image_paths[index]), self.labels[index]

    def _slice(self, shallow_copy, slice_range):
        shallow_copy.image_paths = self.image_paths[slice_range]
        shallow_copy.labels = self.labels[slice_range]

    def __len__(self):
        return len(self.image_paths)

    def unzip(self):
        img_buff = ImageBuffer(self.image_paths, self.loader)
        return img_buff, self.labels
        
