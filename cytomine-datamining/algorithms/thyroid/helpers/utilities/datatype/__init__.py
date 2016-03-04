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


from .bounds import Bounds
from .polygon import bounds, crop_polygon, affine_transform
from .tile import Tile
from .imageconverter import OverflowManager, ClipOverflow, HistogramEqualizer
from .imageconverter import ImageConverter, NumpyConverter, PILConverter
from .imageconverter import CVConverter


__all__ = ["Tile", "Bounds", "bounds", "crop_polygon",
		   "ImageConverter", "OverflowManager", "ClipOverflow", "NumpyConverter",
		   "HistogramEqualizer", "PILConverter", "CVConverter", "affine_transform"]
