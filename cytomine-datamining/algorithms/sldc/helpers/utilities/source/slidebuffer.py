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

from ..indexer import Sliceable


class SlideBuffer(Sliceable):
    """
    ===========
    SlideBuffer
    ===========
    An :class:`SlideBuffer` is a buffer of predefined length which returns
    :class:`TileStream`. Each seed refers to a new slide.

    The [int] operator returns a :class:`TileStream`
    The [int:int] operator returns an :class:`SlideBuffer`

    Constructor parameters
    ----------------------
    seed_sequence : iterable of objects
        A seed contains all the information required to load the image it spans
    tile_stream_builder : TileStreamBuilder (default : UniqueTileStreamBuilder)
        A builder for :class:`TileStream`s
    """

    def __init__(self, seed_sequence, tile_stream_builder):
        self.seeds = seed_sequence
        self.builder = tile_stream_builder

    def _get(self, index):
        return self.builder.build(self.seeds[index])

    def _slice(self, shallow_copy, slice_range):
        shallow_copy.seeds = self.seeds[slice_range]

    def __len__(self):
        return len(self.seeds)
