# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


def emplace(src, dest, mapping):
    """Place the values or src into dest at the indexes indicated by the mapping

    Parameters
    ----------
    src: list (size: n)
        Elements to emplace into the dest list
    dest: list (size: m)
        The list in which the elements of src must be placed
    mapping: list (size: n)
        The indexes of dest where the elements of src must be placed
    """
    for index, value in zip(mapping, src):
        dest[index] = value


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
        prediction: int|None
            An integer code indicating the predicted class.
            If none of the dispatching rules matched a polygon, the value returned is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered,
            None is returned.
        """
        matching_rule = self._dispatch_index(polygon)
        if matching_rule == -1:
            return self._fail_callback(polygon)
        else:
            return self._classifiers[matching_rule].predict(image, polygon)

    def dispatch_classify_batch(self, image, polygons):
        """Apply the dispatching and classification steps to an ensemble of polygons.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygons: list of shapely.geometry.Polygon
            The polygons of which the classes must be predicted

        Returns
        -------
        predictions: list of int|None
            A list of integer codes indicating the predicted classes.
            If none of the dispatching rules matched the polygon, the prediction associated with it is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered, None is
            returned.
        """
        match_dict = dict()  # maps rule indexes with matching polygons
        poly_ind_dict = dict()  # maps rule indexes with index of the polygons in the passed array
        for i, polygon in enumerate(polygons):
            match_index = self._dispatch_index(polygon)
            match_dict[match_index] = match_dict.get(match_index, []) + [polygon]
            poly_ind_dict[match_index] = poly_ind_dict.get(match_index, []) + [i]
        predict_list = [None] * len(polygons)
        for index in match_dict.keys():
            if index == -1:
                predictions = [self._fail_callback(polygon) for polygon in match_dict[index]]
            else:
                predictions = self._classifiers[index].predict_batch(image, match_dict[index])
            emplace(predictions, predict_list, poly_ind_dict[index])
        return predict_list

    def _dispatch_index(self, polygon):
        """Return the index of the first rule that matched the polygon. Return -1 if none of the rules matched.
        """
        for i, rule in enumerate(self._predicates):
            if rule.evaluate(polygon):
                return i
        return -1
