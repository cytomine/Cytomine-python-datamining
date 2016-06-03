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

from abc import ABCMeta, abstractmethod
from copy import copy


class Indexable:
    """
    =========
    Indexable
    =========
    An :class:`Indexable` is an object which contains elements that can be
    accessed via a :method:`_get` method specifying the index and has a
    length accessed via the :function:`len` function.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _get(self, index=0):
        """
        Returns the element at the given index

        Parameters
        ----------
        index : int (default : 0)
            The index of the element

        Return
        ------
        The corresponding element
        """
        pass

    def __getitem__(self, index=0):
        """
        Returns the element at the given index

        Parameters
        ----------
        index : int (default : 0)
            The index of the element

        Return
        ------
        The corresponding element
        """
        if index < 0:
            index = len(self) + index
        return self._get(index)

    @abstractmethod
    def __len__(self):
        pass


class Sliceable(Indexable):
    """
    =========
    Sliceable
    =========
    A :class:`Sliceable` is an :class:`Indexable` which can be sliced via
    the [] operator.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def _slice(self, shallow_copy, slice_range):
        """
        Assign the sliced elements to the shallow copy

        Parameters
        ----------
        shallow_copy : object of the same class as self
            The copy on which to attach the sliced elements
        slice_range : slice object
            The slice range
        """
        pass

    def __getitem__(self, index=0):
        """
        Returns either the element at the given index or a slice of this object

        Parameters
        ----------
        index
            int (default : 0)
                The index of the element
            slice
                The slice to extract

        Return
        ------
        The corresponding element or slice
        """
        #If the index is a slice, we return a clone of this object with
        # the sliced pair containers
        if isinstance(index, slice):
            clone = copy(self)
            self._slice(clone, index)
            return clone
        #If it is a real index (int), we return the corresponding object
        else:
            return self._get(index)

