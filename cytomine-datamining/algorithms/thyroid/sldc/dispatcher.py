# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod

from logging import Loggable, SilentLogger

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


def emplace(src, dest, mapping):
    """Place the values of src into dest at the indexes indicated by the mapping

    Parameters
    ----------
    src: iterable (size: n)
        Elements to emplace into the dest list
    dest: list (size: m)
        The list in which the elements of src must be placed
    mapping: iterable (subtype: int, size: n)
        The indexes of dest where the elements of src must be placed
    """
    for index, value in zip(mapping, src):
        dest[index] = value


def take(src, idx):
    """Generate a list containing the elements of src of which the index is contained in idx

    Parameters
    ----------
    src: list (size: n)
        Source iterable from which elements must be taken
    idx: iterable (subtype: int, range: [0, n[, size: m)
        Indexes iterable

    Returns
    -------
    list: list
        The list of taken elements
    """
    return [src[i] for i in idx]


class DispatchingRule(Loggable):
    """An interface to be implemented by any class that defined a dispatching rule for polygons
    """
    __metaclass__ = ABCMeta

    def __init__(self, logger=SilentLogger()):
        """Constructor for DispatchingRule objects
        Parameters
        ----------
        logger: Logger
            A logger
        """
        Loggable.__init__(self, logger)

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
        return self.evaluate_batch(image, [polygon])[0]

    @abstractmethod
    def evaluate_batch(self, image, polygons):
        """Evaluate the polygons

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygons
        polygons: iterable (subtype: shapely.geometry.Polygon)
            The list of polygons to dispatch

        Returns
        -------
        result: iterable (subtype: bool)
            Iterable of which the ith element is True if the ith polygon is evaluated to True, False otherwise
        """
        pass


class CatchAllRule(DispatchingRule):
    """A rule which evaluates all the polygons to True"""
    def evaluate_batch(self, image, polygons):
        return [True] * len(polygons)


class DispatcherClassifier(Loggable):
    """A dispatcher classifier is an object that evaluates a set of dispatching rules on polygons extracted from an
    image and that passes these polygons to their associated polygon classifiers according to rule that matched
    them.
    """

    def __init__(self, rules, classifiers, fail_callback=None, logger=SilentLogger()):
        """Constructor for ClassifierDispatcher object

        Parameters
        ----------
        rules: iterable (subtype: DispatchingRule, size: N)
            An iterable containing DispatchingRule objects implementing the polygon dispatching logic.
        classifiers: iterable (subtype: PolygonClassifiers, size: N)
            An iterable of polygon classifiers associated with the rules.
        fail_callback: callable (optional, default: None)
            A callback to which is passed the polygon if none of the predicates returned True.
            This callback should return the value expected for a polygon which doesn't match any rule.
            If the default value is passed, a custom callable which always returns None is crafted.
        """
        Loggable.__init__(self, logger)
        self._rules = rules
        self._classifiers = classifiers
        self._fail_callback = fail_callback if fail_callback is not None else (lambda x: None)

    def dispatch_classify(self, image, polygon):
        """Dispatch a single polygon to its corresponding classifier according to the dispatching rules,
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
        probability: float
            The probability associated with the prediction (0.0 if the polygon wasn't dispatched)
        dispatch: int
            The index of the rule that matched the polygon, -1 if none did
        """
        matching_rule = self._dispatch_index(image, polygon)
        if matching_rule == -1:
            return self._fail_callback(polygon), 0.0, matching_rule
        else:
            cls, probability = self._classifiers[matching_rule].predict(image, polygon)
            return cls, probability, matching_rule

    def dispatch_classify_batch(self, image, polygons, timing):
        """Apply the dispatching and classification steps to an ensemble of polygons.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygons: iterable (subtype: shapely.geometry.Polygon, size: N)
            The polygons of which the classes must be predicted
        timing: WorkflowTiming
            The timing object for computing the execution times of dispatching and classification

        Returns
        -------
        predictions: iterable (subtype: int|None, size: N)
            A list of integer codes indicating the predicted classes.
            If none of the dispatching rules matched the polygon, the prediction associated with it is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered, None is
            returned.
        probabilities: iterable (subtype: float, range: [0,1], size: N)
            The probabilities associated with each predicted classes (0.0 for polygons that were not dispatched)
        dispatches: iterable (subtype: int, size: N)
            A iterable of integers of which the ith one is the index of the rule that matched the ith polygon.
        """
        match_dict = dict()  # maps rule indexes with matching polygons
        poly_ind_dict = dict()  # maps rule indexes with index of the polygons in the passed array
        indexes = np.arange(len(polygons))
        # check which rule matched the polygons
        for i, rule in enumerate(self._rules):
            if indexes.shape[0] == 0:  # if there are no more elements to evaluate
                break
            match, no_match = self._split_by_rule(image, rule, polygons, indexes, timing)
            if len(match) > 0:  # skip if there are no match
                match_dict[i] = match_dict.get(i, []) + take(polygons, match)
                poly_ind_dict[i] = poly_ind_dict.get(i, []) + list(match)
                indexes = np.setdiff1d(indexes, match, True)

        # log the end of dispatching
        nb_polygons = len(polygons)
        nb_dispatched = nb_polygons - indexes.shape[0]
        self.logger.info("DispatcherClassifier : end dispatching ({}/{} polygons dispatched).".format(nb_dispatched,
                                                                                                      nb_polygons))

        # add all polygons that didn't match any rule
        match_dict[-1] = take(polygons, indexes)
        poly_ind_dict[-1] = indexes

        # compute the prediction
        predict_list = [None] * len(polygons)
        probabilities_list = [0.0] * len(polygons)
        dispatch_list = [None] * len(polygons)
        for index in match_dict.keys():
            if index == -1:
                # set a 0 probability for each poly as they were not dispatched
                probabilities = np.zeros((len(match_dict[index]),))
                predictions = [self._fail_callback(polygon) for polygon in match_dict[index]]
            else:
                timing.start_classify()
                predictions, probabilities = self._classifiers[index].predict_batch(image, match_dict[index])
                timing.end_classify()
            # Emplace prediction in prediction list, probabilities in probabilities
            # list and dispatch id in dispatch list
            emplace(predictions, predict_list, poly_ind_dict[index])
            emplace(probabilities, probabilities_list, poly_ind_dict[index])
            emplace(np.full((len(predictions),), index, dtype="int"), dispatch_list, poly_ind_dict[index])

        self.logger.info("DispatcherClassifier : end classification.")
        return predict_list, probabilities_list, dispatch_list

    def _split_by_rule(self, image, rule, polygons, poly_indexes, timing):
        """Given a rule, splits all the poly_indexes list into two lists. The first list contains
        the indexes corresponding to the polygons that were evaluated True by the rule, the indexes that
        were evaluated False by the rule.

        Parameters
        ----------
        image: Image
            The image from which were extracted the polygons
        rule: DispatchingRule
            The rule with which the polygons must be evaluated
        polygons: iterable
            The list of polygons
        poly_indexes: iterable
            The indexes of the polygons from the list polygons to process
        timing: WorkflowTiming
            The timing object for computing the dispatching time
        Returns
        -------
        true_list: iterable (subtype: int)
            The indexes that were evaluated true
        false_list: iterable (subtype: int)
            The indexes that were evaluated false
        """
        polygons_to_evaluate = take(polygons, poly_indexes)
        timing.start_dispatch()
        eval_results = rule.evaluate_batch(image, polygons_to_evaluate)
        timing.end_dispatch()
        np_indexes = np.array(poly_indexes)
        return np_indexes[np.where(eval_results)], np_indexes[np.where(np.logical_not(eval_results))]

    def _dispatch_index(self, image, polygon):
        """Return the index of the first rule that matched the polygon. Return -1 if none of the rules matched.
        """
        for i, rule in enumerate(self._rules):
            if rule.evaluate(image, polygon):
                return i
        return -1
