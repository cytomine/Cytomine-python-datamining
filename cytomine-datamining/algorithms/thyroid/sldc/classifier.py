# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class PolygonClassifier(object):
    """A classifier that classifies polygons
    """
    __metaclass__ = ABCMeta

    def predict(self, image, polygon):
        """Predict the class associated with the given polygon

        Parameters
        ----------
        image: Image
            The image the object of interest delimited by the polygon
        polygon: shapely.geometry.Polygon
            The polygon of which the class must be predicted

        Returns
        -------
        prediction: int
            An integer code indicating the predicted class
        probability: float (in [0,1])
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
            The image the objects of interest delimited by the polygons
        polygons: iterable (subtype: shapely.geometry.Polygon, size: N)
            The polygons of which the classes must be predicted

        Returns
        -------
        predictions: iterable (subtype: int, size: N)
            An iterable of integer codes indicating the predicted classes
        probabilities: iterable (subtype: int, range: [0,1], size: N)
            The probabilities associated with the classes predicted for each polygon
        """
        pass
