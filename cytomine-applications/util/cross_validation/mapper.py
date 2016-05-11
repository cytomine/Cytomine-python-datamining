# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class ClassificationOutputMapper(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, term):
        """
        Map a term with the actual class
        :param term: int, the term of which the actual class is desired
        :return: int, the actual class
        """
        pass

    def map_dict(self, terms):
        unique_terms = np.unique(np.array(terms))
        map_dict = dict()
        for term in unique_terms.tolist():
            mapped = self.map(term)
            map_dict[mapped] = map_dict.get(mapped, []) + [term]
        return map_dict


class DefaultMapper(ClassificationOutputMapper):
    def map(self, term):
        return term


class BinaryMapper(ClassificationOutputMapper):
    def __init__(self, positive_classes, negative_classes):
        self._positive_classes = set(positive_classes)
        self._negative_classes = set(negative_classes)

    def map(self, term):
        if term in self._positive_classes:
            return 1
        elif term in self._negative_classes:
            return 0
        else:
            raise ValueError("Unexpected term : {} ({}).".format(term, type(term)))


class TernaryMapper(BinaryMapper):
    def __init__(self, positive_classes, negative_classes, other_classes):
        BinaryMapper.__init__(self, positive_classes, negative_classes)
        self._other_classes = other_classes

    def map(self, term):
        if term in self._other_classes:
            return 2
        return super(TernaryMapper, self).map(term)
