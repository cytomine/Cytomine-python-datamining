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


class Bounds(object):
    """
    ======
    Bounds
    ======
    A class representing integer bounding box. Its elements are accessible with
    the [] operator. The accepted keys are :
    'x' : the minimum x
    'y' : the minimum y
    'w' : the width
    'h' : the height

    Example
    -------

    >>> bounds = Bounds(3, 4, 5, 6)
    >>> bounds['x']
    3
    >>> bounds['y']
    4
    >>> bounds['w']
    5
    >>> bounds['h']
    6
    >>> x, y, width, height = bounds
    >>> bounds.y == y
    True
    


    Constructor parameters
    ----------------------
    x : int
        The minimum x
    y : int 
        The minimum y
    width : int
        The width of the bounding box
    height : int
        The height of the bounding box
    """

    def __init__(self, x, y, width, height):
        self._dict = dict()
        self._dict["x"] = x
        self._dict["y"] = y
        self._dict["w"] = width
        self._dict["h"] = height

    @property
    def x(self):
        return self._dict["x"]

    @x.setter
    def x(self, x):
        self._dict["x"] = x

    @property
    def y(self):
        return self._dict["y"]

    @y.setter
    def y(self, y):
        self._dict["y"] = y

    @property
    def width(self):
        return self._dict["w"]

    @width.setter
    def width(self, width):
        self._dict["width"] = width


    @property
    def height(self):
        return self._dict["h"]

    @height.setter
    def height(self, height):
        self._dict["height"] = height


    def __iter__(self):
        yield self.x
        yield self.y
        yield self.width
        yield self.height


    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return "(%d, %d, %d, %d)" % (self._dict['x'], self._dict['y'], 
                                     self._dict['w'], self._dict['h'])


