# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

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
