# -*- coding: utf-8 -*-
from abc import ABCMeta

import numpy as np

from sldc import DispatchingRule, SilentLogger
from classifiers import PyxitClassifierAdapter

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def check_area(resolution, polygons, min_area):
    """Return an array of boolean where the ith element is True if the ith polygon has an area greated than min_area

    Parameters
    ----------
    resolution: float
        The resolution of the image
    polygons: iterable (subtype: shapely.geometry.Polygon)
        The polygons
    min_area: float
        The minimum area

    Returns
    -------
    is_greater: ndarray
        Array of boolean
    """
    resolution = resolution if resolution is not None else 1
    return (resolution * resolution * np.array([polygon.area for polygon in polygons])) > min_area


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
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger(), min_area=15.0):
        """Constructor for AggregateRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for aggregate and 0 for
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        min_area: float (optional, default: 15.0)
            The minimum area of an accepted object (in µm²)
        """
        ClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)
        self._min_area = min_area

    def evaluate_batch(self, image, polygons):
        min_area_checked = check_area(image.image_instance.resolution, polygons, self._min_area)
        classes, proba = self._classifier.predict_batch(image, np.array(polygons)[min_area_checked])
        all_classes = np.full((len(polygons),), 1, dtype="int")
        all_classes[min_area_checked] = np.array(classes)
        return all_classes < 0.5


class CellRule(ClassifierRule):
    def __init__(self, pyxit_classifier_adapter, logger=SilentLogger(), min_area=15.0):
        """Constructor for CellRule object

        Parameters
        ----------
        pyxit_classifier_adapter: PyxitClassifierAdapter
            A pyxit binary classifier predicting 1 for cells and 0 for others
        logger: Logger (optional, default: a SilentLogger instance)
            A logger object
        min_area: float (optional, default: 15.0)
            The minimum area of an accepted object (in µm²)
        """
        ClassifierRule.__init__(self, pyxit_classifier_adapter, logger=logger)
        self._min_area = min_area

    def evaluate_batch(self, image, polygons):
        min_area_checked = check_area(image.image_instance.resolution, polygons, self._min_area)
        classes, proba = self._classifier.predict_batch(image, np.array(polygons)[min_area_checked])
        all_classes = np.zeros((len(polygons),), dtype="int")
        all_classes[min_area_checked] = np.array(classes)
        return np.logical_and(all_classes > 0.5, all_classes < 1.5)


class CellGeometricRule(DispatchingRule):
    """Cell """
    def __init__(self, min_area=15.0, min_circularity=0.6):
        """Constructor

        Parameters
        ----------
        min_area: float (optional, default: 15.0)
            Minimum area for a cell (in µm²)
        min_circularity: float (optional, default: 0.6)
            Minimum circularity for a cell
        """
        DispatchingRule.__init__(self)
        self._min_area = min_area
        self._min_circularity = min_circularity

    def evaluate_batch(self, image, polygons):
        resolution = image.image_instance.resolution if image.image_instance.resolution is not None else 1
        areas = np.array([polygon.area for polygon in polygons]) * resolution * resolution
        perim = np.array([polygon.length for polygon in polygons]) * resolution
        circ = 4 * np.pi * areas / (perim * perim)
        return np.logical_and(areas > self._min_area, circ > self._min_circularity)
