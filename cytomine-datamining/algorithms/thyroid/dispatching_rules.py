# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from sldc import DispatchingRule
from classifiers import PyxitClassifierAdapter


class BinaryClassifierRule(DispatchingRule):
    def __init__(self, pyxit_classifier_adapter):
        """Constructor for BinaryClassifierRule object
        The evaluate returns true if the classifier predicts 1, it returns false otherwise

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier
        """
        self._classifier = pyxit_classifier_adapter

    def evaluate(self, image, polygon):
        return self._classifier.predict(image, polygon) > 0.5

    def evaluate_batch(self, image, polygons):
        return self._classifier.predict_batch(image, polygons) > 0.5


class AggregateRule(BinaryClassifierRule):
    def __init__(self, pyxit_classifier_adapter):
        """Constructor for AggregateRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for aggregate and 0 for others
        """
        BinaryClassifierRule.__init__(self, pyxit_classifier_adapter)


class CellRule(BinaryClassifierRule):
    def __init__(self, pyxit_classifier_adapter):
        """Constructor for CellRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for cells and 0 for others
        """
        BinaryClassifierRule.__init__(self, pyxit_classifier_adapter)
