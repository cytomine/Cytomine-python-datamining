# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'

import numpy as np
# Direct import of Image works with ImageDraw, indirect (PIL.Image) does not work
try:
    import Image, ImageDraw
except:
    from PIL import Image, ImageDraw
from ..utilities.datatype import crop_polygon

#TODO make note of the rasterization colors
class AbstractRasterizer:
    """
    ==================
    AbstractRasterizer
    ==================
    Abstract base class for rasterizer
    """

    def rasterize(self, polygon):
        """
        Rasterize the given polygon. The polygon is assumed to be expressed
        in a traditional (i.e. lower left) coordinate system.

        Parameters
        ----------
        polygon : :class:`shapely.Polygon`
            The polygon to rasterize. The polygon is assumed to be in
            (x, y)-coordinate system where x is the column of the image
            on which to rasterize (increasing rightwards) and y is the row
            (increasing downwards)


        Return
        ------
        np_image : numpy.ndarray
            The image (in numpy format) of the rasterization of the polygon.
            The image should have the same dimension as the bounding box of
            the polygon.
        """
        pass

    def alpha_rasterize(self, image, polygon):
        """
        Rasterize the given polygon as an alpha mask of the given image.
        The polygon should fit inside the image (its width and heights are
        smaller than those of the image). The polygon is assumed to be
        expressed in a traditional (i.e. lower left) coordinate system and
        will be translated so that the left most point will be on the first
        column and the top most on the first row.

        Parameters
        ----------
        polygon : :class:`shapely.Polygon`
            The polygon to rasterize

        Return
        ------
        np_image : numpy.ndarray
            The image (in numpy format) of the rasterization of the polygon.
            The image should have the same dimension as the bounding box of
            the polygon.
        """
        pass


class Rasterizer:
    """
    ==========
    Rasterizer
    ==========
    A concrete implementation of :class:`AbstractRasterizer`
    """

    def rasterize(self, polygon):
        polygon = crop_polygon(polygon)
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx + 1
        height = maxy - miny + 1
        img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(img)
        seq_pts = polygon.boundary.coords
        draw.polygon(seq_pts, outline=255, fill=255)
        return np.asarray(img)

    def alpha_rasterize(self, image, polygon):
        #Creating holder
        np_img = np.asarray(image)
        width, height, depth = np_img.shape
        depth += 1
        np_results = np.zeros((width, height, depth), dtype=np.uint)
        np_results[:, :, 0:depth-1] = np_img
        #Rasterization
        polygon = crop_polygon(polygon)
        alpha = Image.new("L", (width, height), 255)
        draw = ImageDraw.Draw(alpha)
        seq_pts = polygon.boundary.coords
        draw.polygon(seq_pts, outline=0, fill=0)
        np_results[:, :, depth-1] = alpha
        return np_results
