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


from .imagebuffer import ImageBuffer, ImageLoader, PILLoader, NotCMapPILLoader
from .imagebuffer import  NumpyLoader
from .slidebuffer import SlideBuffer
from .tilestream import TileStream, UniqueTileStream, TileStreamBuilder

__all__ = ["ImageBuffer", "ImageLoader", "PILLoader", "NotCMapPILLoader", 
		   "NumpyLoader","SlideBuffer", "TileStream", "UniqueTileStream", 
		   "TileStreamBuilder"]

