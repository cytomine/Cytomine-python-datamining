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
from ..datatype.tile import Tile


class TileStream:
    """
    ===========
    TileStream
    ===========
    An :class:`TileStream` is a stream of tiles belonging to the same
    image. The tiles may be loaded on request and may overlap.
    """
    __metaclass__ = ABCMeta

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def get_image_id(self):
        pass

class UniqueTileStream(TileStream):
    """
    ================
    UniqueTileStream
    ================
    A :class:`UniqueTileStream` is a stream which returns a single tile which
    is a whole image

    Constructor parameters
    ----------------------
    image_loader : :class:`ImageLoader`
        the object which loads the tile
    seed : TileLoader dependant
            A seed to load the image
    """
    def __init__(self, image_loader, seed):
        self.loader = image_loader
        self.seed = seed

    def next(self):
        return Tile(self.loader.load(self.seed))


class TileStreamBuilder:
    """
    =================
    TileStreamBuilder
    =================
    An abstract base class for builder which produces :class:`TileStream`
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, seed):
        """
        Builds a :class:`TileStream`.

        Parameters
        ----------
        seed : TileLoader dependant
            A seed to load the image
        """
        pass
