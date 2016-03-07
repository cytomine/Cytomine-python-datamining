# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class PolygonClassifier(object):
    """
    A classifier for polygons of an image
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, image, polygon):
        """Predict the class associated with the given polygon

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygon: shapely.geometry.Polygon
            The polygon of which the class must be predicted

        Returns
        -------
        prediction: int
            An integer code indicating the predicted class
        """
        pass

    def predict_batch(self, image, polygons):
        """Predict the classes associated with the given polygons

        Parameters
        ----------
        image: Image
            The image to which belongs the polygons
        polygons: list of shapely.geometry.Polygon
            The polygons of which the class must be predicted

        Returns
        -------
        predictions: list of int
            A list of integer codes indicating the predicted classes

        Note
        ----
        Default implementation simply loops over the polygons and call predict(image, polygons[i])
        """
        return [self.predict(image, polygon) for polygon in polygons]
