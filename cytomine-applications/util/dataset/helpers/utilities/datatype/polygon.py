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

from functools import partial

from .bounds import Bounds
from shapely.affinity import affine_transform as aff_transfo
from shapely.geometry import box

def bounds(polygon):
    """
    Factory class method which returns the (integer) bounds
    of a polygon

    Parameters
    ----------
    polygon : :class:`shapely.Polygon`
        The polygon whose bounds to compute

    Return
    ------
    bounds : :class:`Bounds`
        The bounds of the polygon
    """
    minx, miny, maxx, maxy = polygon.bounds
    minx = int(minx)
    miny = int(miny)
    maxx = int(maxx)
    maxy = int(maxy)
    width = maxx - minx + 1
    height = maxy - miny + 1
    return Bounds(minx, miny, width, height)


def clamp_polygon(polygon, minx, miny, maxx, maxy):
    b = box(minx,miny, maxx, maxy)
    return b.intersection(polygon)


def affine_transform(xx_coef=1, xy_coef=0, yx_coef=0, yy_coef=1,
                     delta_x=0, delta_y=0):
    """
    Represents a 2D affine trasnformation:

    x' = xx_coef * x + xy_coef * y + delta_x
    y' = yx_coef * x + yy_coef * y + delta_y

    Constructor parameters
    ----------------------
    xx_coef : float (default : 1)
        The x from x coefficient
    xy_coef : float (default : 0)
        The x from y coefficient
    yx_coef : float (default : 0)
        The y from x coefficient
    yy_coef : float (default : 1)
        The y from y coefficient
    delta_x : float (default : 0)
        The translation over x-axis
    delta_y : float (default : 0)
        The translation over y-axis

    Return
    ------
    affine_transformer : callable: :class:`Shapely.Geometry` => :class:`Shapely.Geometry`
        The function representing the 2D affine transformation
    """
    return partial(aff_transfo, matrix=[xx_coef, xy_coef, yx_coef, yy_coef,
                                        delta_x, delta_y])



