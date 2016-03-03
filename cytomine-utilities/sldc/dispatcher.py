# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class DispatchingRule(object):
    """
    An object for dispatching polygons
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, polygon):
        """Evaluate a polygon

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            The polygon to evaluate

        Returns
        -------
        result: bool
            True if the dispatching rule matched the polygon, False otherwise.
        """
        pass


class DispatcherClassifier(object):
    """
    A dispatcher classifier is an object that evaluates a set of dispatching rules on a polygon
    and that passes this polygon to the associated polygon classifier according to
    the first matching rule.
    """

    def __init__(self, predicates, classifiers, fail_callback=None):
        """Constructor for ClassifierDispatcher object

        Parameters
        ----------
        predicates: list of DispatchingRule objects, size : N
            A list of ordered DispatchingRule implementing the polygon dispatching logic.
        classifiers: list of PolygonClassifiers objects, size : N
            A list of classifiers associated with the predicates.
        fail_callback: callable, optional (default: None)
            A callback to which is passed the polygon if none of the predicates returns True.
            This callback should return the value expected for a polygon which doesn't match any rule.
            If the default value is passed, a custom callable which always returns None is crafted.
        """
        self._predicates = predicates
        self._classifiers = classifiers
        self._fail_callback = fail_callback if fail_callback is not None else (lambda x: None)

    def dispatch_classify(self, image, polygon):
        """Dispatch the polygon to its corresponding classifier according to the dispatching rules,
        then compute and return the associated prediction.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygon: shapely.geometry.Polygon
            The polygon of which the class must be predicted

        Returns
        -------
        prediction: int|mixed
            An integer code indicating the predicted class.
            If none of the dispatching rule matches the polygon, the value returned is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered,
            None is returned.
        """
        for rule, classifier in zip(self._predicates, self._classifiers):
            if rule.evaluate(polygon):
                return classifier.predict(image, polygon)
        return self._fail_callback(polygon)
