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


class Iterable2D:
    def get_height(self):
        pass

    def get_width(self):
        pass

    def get_element(self, row, col):
        pass


class RowOrderIterator:

    def __init__(self, iterable2d):
        self.iterable = iterable2d
        self.col = -1
        self.row = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.col += 1
        if self.col >= self.iterable.get_width():
            self.col = 0
            self.row += 1
        if self.row >= self.iterable.get_height():
            raise StopIteration()
        return self.iterable.get_element(self.row, self.col)


class RowPrimeIterator:

    def __init__(self):
        raise NotImplementedError("Not yet implemented")


class CantorDiagonalIterator:
    def __init__(self):
        raise NotImplementedError("Not yet implemented")


class MortonIterator:
    def __init__(self):
        raise NotImplementedError("Not yet implemented")


class HilbertIterator:
    def __init__(self):
        raise NotImplementedError("Not yet implemented")
