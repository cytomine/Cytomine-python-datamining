# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


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


def take(src, idx):
    """Generate a list containing the elements of src of which the index is contained in idx

    Parameters
    ----------
    src: list (size: n)
        Source list from which elements must be taken
    idx: list (size: m)
        Index list (indexes should be integers in [0, n[

    Returns
    -------
    list: list
        The list of taken elements
    """
    return [src[i] for i in idx]


class DispatchingRule(object):
    """
    An object for dispatching polygons
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, image, polygon):
        """Evaluate a polygon

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygon
        polygon: shapely.geometry.Polygon
            The polygon to evaluate

        Returns
        -------
        result: bool
            True if the dispatching rule matched the polygon, False otherwise.
        """
        pass

    def evaluate_batch(self, image, polygons):
        """Evaluate the polygons

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygons
        polygons: list
            The list of polygons to dispatch

        Returns
        -------
        result: list of bool
            List of which the ith element is True if the ith polygon is evaluated to True, False otherwise
        """
        return [self.evaluate(image, polygon) for polygon in polygons]


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
        dispatch: int
            The index of the dispatch rule that matched the polygon, -1 if none did
        """
        matching_rule = self._dispatch_index(image, polygon)
        if matching_rule == -1:
            return self._fail_callback(polygon), matching_rule
        else:
            return self._classifiers[matching_rule].predict(image, polygon), matching_rule

    def dispatch_classify_batch(self, image, polygons, logger):
        """Apply the dispatching and classification steps to an ensemble of polygons.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygons: list of shapely.geometry.Polygon
            The polygons of which the classes must be predicted
        logger: logger
            A logger
        Returns
        -------
        predictions: list of int|None
            A list of integer codes indicating the predicted classes.
            If none of the dispatching rules matched the polygon, the prediction associated with it is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered, None is
            returned.
        dispatches: list of int
            A list of integers of which the ith one is the index of first rule that matched the ith polygon.
        """
        match_dict = dict()  # maps rule indexes with matching polygons
        poly_ind_dict = dict()  # maps rule indexes with index of the polygons in the passed array
        indexes = np.arange(len(polygons))
        # check which rule matched the polygons
        for i, rule in enumerate(self._predicates):
            if indexes.shape[0] == 0:  # if there are no more elements to evaluate
                break
            match, no_match = self._split_by_rule(image, rule, polygons, indexes)
            if len(match) > 0: # skip if there are no match
                match_dict[i] = match_dict.get(i, []) + take(polygons, match)
                poly_ind_dict[i] = poly_ind_dict.get(i, []) + list(match)
                indexes = np.setdiff1d(indexes, match, True)

        # log the end of dispatching
        nb_polygons = len(polygons)
        nb_dispatched = nb_polygons - indexes.shape[0]
        logger.info("DispatcherClassifier : end dispatching ({}/{} polygons dispatched)".format(nb_dispatched, nb_polygons))

        # add all polygons that didn't match any rule
        match_dict[-1] = take(polygons, indexes)
        poly_ind_dict[-1] = indexes

        # compute the prediction
        predict_list = [None] * len(polygons)
        dispatch_list = [None] * len(polygons)
        for index in match_dict.keys():
            if index == -1:
                predictions = [self._fail_callback(polygon) for polygon in match_dict[index]]
            else:
                predictions = self._classifiers[index].predict_batch(image, match_dict[index])
            # emplace prediction in prediction list
            emplace(predictions, predict_list, poly_ind_dict[index])
            # emplace dispatch id in dispatch list
            emplace(np.full((len(predictions),), index).astype('int'), dispatch_list, poly_ind_dict[index])

        logger.info("DispatcherClassifier : end dispatching ({}/{} polygons dispatched)".format(nb_dispatched, nb_polygons))
        return predict_list, dispatch_list

    def _split_by_rule(self, image, rule, polygons, poly_indexes):
        """Given a rule, splits all the poly_indexes list into two lists. The first list contains
        the indexes corresponding to the polygons that were evaluated True by the rule, the indexes that
        were evaluated False by the rule.

        Parameters
        ----------
        image: Image
            The image from which were extracted the polygons
        rule: DispatchingRule
            The rule with which the polygons must be evaluated
        polygons: list of Polygon
            The list of polygons
        poly_indexes: list of int
            The indexes of the polygons from the list polygons to process

        Returns
        -------
        true_list: list of int
            The indexes that were evaluated true
        false_list: list of int
            The indexes that were evaluated false
        """
        polygons_to_evaluate = take(polygons, poly_indexes)
        eval_results = rule.evaluate_batch(image, polygons_to_evaluate)
        np_indexes = np.array(poly_indexes)
        return np_indexes[np.where(eval_results)], np_indexes[np.where(np.logical_not(eval_results))]

    def _dispatch_index(self, image, polygon):
        """Return the index of the first rule that matched the polygon. Return -1 if none of the rules matched.
        """
        for i, rule in enumerate(self._predicates):
            if rule.evaluate(image, polygon):
                return i
        return -1
