# -*- coding: utf-8 -*-
import pickle

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

import os
import numpy as np
from PIL.Image import fromarray
from sldc import PolygonClassifier


class PyxitClassifierAdapter(PolygonClassifier):

    def __init__(self, pyxit_classifier, tile_builder, classes, working_path):
        """Constructor for PyxitClassifierAdapter objects

        Parameters
        ----------
        pyxit_classifier: PyxitClassifier
            The pyxit classifier objects
        tile_builder: TileBuilder
            A tile builder
        classes: array
            An array containing the classes labels
        working_path: string
            A path in which the instance can save temporary images to pass to the pyxit classifier
        """
        self._pyxit_classifier = pyxit_classifier
        self._tile_builder = tile_builder
        self._classes = classes
        self._working_path = working_path

    def predict(self, image, polygon):
        # Pyxit classifier takes images from the filesystem
        # So store the crop into a file before passing the path to the classifier
        tile, tile_path = self._extract_tile(image, polygon)
        np_image = tile.np_image
        fromarray(np_image).save(tile_path)
        return self._predict(np.array([tile_path]))[0]

    def predict_batch(self, image, polygons):
        # Pyxit classifier takes images from the filesystem
        # So store the crops into files before passing the paths to the classifier
        paths = list()
        for i, polygon in enumerate(polygons):
            tile, tile_path = self._extract_tile(image, polygon)
            np_image = tile.np_image
            fromarray(np_image).save(tile_path)
            paths.append(tile_path)
        return self._predict(np.array(paths))

    def _predict(self, X):
        """Call predict on the classifier with the filepaths of X

        Parameters
        ----------
        X: list of string
            The path to the image to classify

        Returns
        -------
        Predictions:
            Return
        """
        probas = self._pyxit_classifier.predict_proba(X)
        best_index = np.argmax(probas, axis=1)
        return self._classes.take(best_index, axis=0)

    def _extract_tile(self, image, polygon):
        """Given an image and a polygon, extract the crop tile and tile path

        Parameters
        ----------
        image: Image
            An image object
        polygon: shapely.geometry.Polygon
            A polygon fitting in the image and for which the crop must be extracted
        Returns
        -------
        tile: Tile
            The crop tile
        tile_path: string
            The path in which should be stored the crop for learning
        """
        minx, miny, maxx, maxy = polygon.bounds
        offset = (minx, miny)
        width = int(maxx - minx) + 1
        height = int(maxy - miny) + 1
        tile = image.tile(self._tile_builder, offset, width, height)
        tile_path = self._tile_path(image, offset, width, height)
        return tile, tile_path

    def _tile_path(self, image, offset, width, height):
        """Return the path where to store the tile

        Parameters
        ----------
        image: Image
            The image object from which the tile was extracted
        offset: tuple (int, int)
            The coordinates of the upper left pixel of the tile
        width: int
            The tile width
        height: int
            The tile height

        Returns
        -------
        path: string
            The path in which to store the image
        """
        filename = "{}_{}_{}_{}_{}.png".format(image.image_instance.id, offset[0], offset[1], width, height)
        return os.path.join(self._working_path, filename)

    @staticmethod
    def build_from_pickle(model_path, tile_builder, working_path):
        """Builds a PyxitClassifierAdapter object from a pickled model

        Parameters
        ----------
        model_path: string
            The path to which is stored the pickled model
        tile_builder: TileBuilder
            A tile builder object
        working_path: string
            The path in which temporary files can be written

        Returns
        -------
        adapter: PyxitClassifierAdapter
            The built classifier adapter

        Notes
        -----
        The first object pickled in the file in the path 'model_path' is an array
        containing the classes, and the second is the PyxitClassifier object
        """
        with open(model_path, "rb") as model_file:
            classes = pickle.load(model_file)
            classifier = pickle.load(model_file)
        return PyxitClassifierAdapter(classifier, tile_builder, classes, working_path)
