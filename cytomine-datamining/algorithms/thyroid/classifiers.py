# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sldc import PolygonClassifier

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class PyxitClassifierAdapter(PolygonClassifier):

    def __init__(self, pyxit_classifier, tile_cache, classes, working_path):
        """Constructor for PyxitClassifierAdapter objects

        Parameters
        ----------
        pyxit_classifier: PyxitClassifier
            The pyxit classifier objects
        tile_cache: TileCache
            A tile cache for fetching tiles
        classes: array
            An array containing the classes labels
        working_path: string
            A path in which the instance can save temporary images to pass to the pyxit classifier
        """
        self._pyxit_classifier = pyxit_classifier
        self._tile_cache = tile_cache
        self._classes = classes
        self._working_path = working_path

    def predict(self, image, polygon):
        # Pyxit classifier takes images from the filesystem
        # So store the crop into a file before passing the path to the classifier
        _, tile_path = self._tile_cache.save_tile(image, polygon, self._working_path, alpha=True)
        return self._predict(np.array([tile_path]))[0]

    def predict_batch(self, image, polygons):
        # Pyxit classifier takes images from the filesystem
        # So store the crops into files before passing the paths to the classifier
        paths = []
        for polygon in polygons:
            _, tile_path = self._tile_cache.save_tile(image, polygon, self._working_path, alpha=True)
            paths.append(tile_path)
        return self._predict(np.array(paths))

    def _predict(self, X):
        """Call predict on the classifier with the filepaths of X

        Parameters
        ----------
        X: list of string
            The path to the images to classify

        Returns
        -------
        Predictions:
            Return
        """
        probas = self._pyxit_classifier.predict_proba(X)
        best_index = np.argmax(probas, axis=1)
        return self._classes.take(best_index, axis=0).astype('int')

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
            classifier.n_jobs = 1
            classifier.verbose = 0
        return PyxitClassifierAdapter(classifier, tile_builder, classes, working_path)
