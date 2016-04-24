# -*- coding: utf-8 -*-

from sldc import DispatchingRule, SilentLogger
from classifiers import PyxitClassifierAdapter

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class BinaryClassifierRule(DispatchingRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for BinaryClassifierRule object
        The evaluate returns true if the classifier predicts 1, it returns false otherwise

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        DispatchingRule.__init__(self, logger=logger)
        self._classifier = pyxit_classifier_adapter

    def evaluate(self, image, polygon):
        return self._classifier.predict(image, polygon) > 0.5

    def evaluate_batch(self, image, polygons):
        return self._classifier.predict_batch(image, polygons) > 0.5


class AggregateRule(BinaryClassifierRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for AggregateRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for aggregate and 0 for
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        BinaryClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)


class CellRule(BinaryClassifierRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger()):
        """Constructor for CellRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for cells and 0 for others
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        """
        BinaryClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)
