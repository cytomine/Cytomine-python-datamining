# -*- coding: utf-8 -*-


class TileExtractionError(RuntimeError):
    """
    Thrown when a tile is requested but cannot be fetched
    """
    pass


class ImageExtractionError(RuntimeError):
    """
    Thrown when an image is requested cannot be extracted
    """
    pass
