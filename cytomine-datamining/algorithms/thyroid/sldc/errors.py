# -*- coding: utf-8 -*-


class TileExtractionException(Exception):
    """
    Thrown when a tile is requested but cannot be fetched
    """
    pass


class ImageExtractionException(Exception):
    """
    Thrown when an image is requested cannot be extracted
    """
    pass
