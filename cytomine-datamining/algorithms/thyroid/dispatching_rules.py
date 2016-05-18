# -*- coding: utf-8 -*-
from abc import ABCMeta

import numpy as np

from sldc import DispatchingRule, SilentLogger
from classifiers import PyxitClassifierAdapter

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class ClassifierRule(DispatchingRule):
    """"""
    __metaclass__ = ABCMeta

    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for ClassifierRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit classifier
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        DispatchingRule.__init__(self, logger=logger)
        self._classifier = pyxit_classifier_adapter


class AggregateRule(ClassifierRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for AggregateRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for aggregate and 0 for
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        ClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)

    def evaluate_batch(self, image, polygons):
        classes, proba = self._classifier.predict_batch(image, polygons)
        return classes < 0.5


class CellRule(ClassifierRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for CellRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for cells and 0 for others
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        ClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)

    def evaluate_batch(self, image, polygons):
        classes, proba = self._classifier.predict_batch(image, polygons)
        return np.logical_and(classes > 0.5, classes < 1.5)
