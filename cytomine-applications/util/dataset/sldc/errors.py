# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class TileExtractionException(Exception):
    """Thrown when a tile is requested but cannot be fetched"""
    pass


class ImageExtractionException(Exception):
    """Thrown when an image is requested cannot be extracted"""
    pass


class MissingComponentException(Exception):
    """Thrown when a component is missing for building an object"""
    pass
