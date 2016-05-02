# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class PolygonClassifier(object):
    """
    A classifier for polygons of an image
    """
    __metaclass__ = ABCMeta

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
        proba: float (in [0,1])
            The prediction probability
        """
        pred, proba = self.predict_batch(image, [polygon])
        return pred[0], proba[0]

    @abstractmethod
    def predict_batch(self, image, polygons):
        """Predict the classes associated with the given polygons

        Parameters
        ----------
        image: Image
            The image to which belongs the polygons
        polygons: list of shapely.geometry.Polygon (size N)
            The polygons of which the class must be predicted

        Returns
        -------
        predictions: list of int (size N)
            A list of integer codes indicating the predicted classes
        probas: list of float (in [0,1]) (size N)
            The probabilities associated with the class predicted for each polygon
        """
        pass
