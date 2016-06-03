# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""
from shapely.geometry.base import BaseMultipartGeometry

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'

import numpy as np
# Direct import of Image works with ImageDraw, indirect (PIL.Image) does not work
try:
    import Image, ImageDraw
except:
    from PIL import Image, ImageDraw


def alpha_rasterize(image, polygon):
    """
    Rasterize the given polygon as an alpha mask of the given image. The
    polygon is assumed to be expressed in a traditional (i.e. lower left)
    coordinate system and will be translated so that the left most point
    will be on the first column and the top most on the first row.

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
    # Creating holder
    np_img = np.asarray(image)
    height, width, depth = np_img.shape
    # if there is already an alpha mask, replace it
    if depth == 4 or depth == 2:
        np_img = np_img[:, :, 0:depth-1]
    else:
        depth += 1
    np_results = np.zeros((height, width, depth), dtype=np.uint)
    np_results[:, :, 0:depth-1] = np_img
    # Rasterization
    alpha = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha)
    boundary = polygon.boundary
    if isinstance(boundary, BaseMultipartGeometry):  # handle multi-part geometries
        for sub_boundary in boundary.geoms:
            seq_pts = sub_boundary.coords
            draw.polygon(seq_pts, outline=0, fill=255)
    else:
        seq_pts = polygon.boundary.coords
        draw.polygon(seq_pts, outline=0, fill=255)
    np_results[:, :, depth-1] = alpha
    return np_results

def rasterize(polygon):
    #polygon = crop_polygon(polygon)
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx + 1
    height = maxy - miny + 1
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    seq_pts = polygon.boundary.coords
    draw.polygon(seq_pts, outline=255, fill=255)
    return np.asarray(img)