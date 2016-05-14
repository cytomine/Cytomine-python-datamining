# -*- coding: utf-8 -*-
import pickle
import numpy as np
from sldc import PolygonClassifier, TileExtractionException, Loggable, SilentLogger, Logger

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class PyxitClassifierAdapter(PolygonClassifier, Loggable):

    def __init__(self, pyxit_classifier, tile_cache, classes, logger=SilentLogger()):
        """Constructor for PyxitClassifierAdapter objects

        Parameters
        ----------
        pyxit_classifier: PyxitClassifier
            The pyxit classifier objects
        tile_cache: TileCache
            A tile cache for fetching tiles
        classes: ndarray
            An array containing the classes labels
        logger: Logger
            A logger object
        """
        PolygonClassifier.__init__(self)
        Loggable.__init__(self, logger)
        self._pyxit_classifier = pyxit_classifier
        self._tile_cache = tile_cache
        self._classes = classes

    def predict_batch(self, image, polygons):
        # Pyxit classifier takes images from the filesystem
        # So store the crops into files before passing the paths to the classifier
        paths = list()
        extracted = list()
        for i, polygon in enumerate(polygons):
            try:
                paths.append(self._tile_cache.polygon_fetch_and_cache(image, polygon))
                extracted.append(i)
            except TileExtractionException as e:
                self.logger.w("PyxitClassifierAdapter: skip polygon because tile cannot be extracted.\n".format(i) +
                              "PyxitClassifierAdapter: error : {}".format(e.message))

        # merge predictions with missed tiles
        predictions, probas = self._predict(np.array(paths))
        ret_classes = [None] * len(polygons)
        ret_probas = [0.0] * len(polygons)
        for prediction, proba, i in zip(predictions, probas, extracted):
            ret_classes[i] = prediction
            ret_probas[i] = np.max(proba)
        return np.array(ret_classes), np.array(ret_probas)

    def _predict(self, X):
        """Call predict on the classifier with the filepaths of X

        Parameters
        ----------
        X: ndarray
            The paths to the images to classify

        Returns
        -------
        predictions: list of int
            The predicted classes
        probas: list of float
            The probabilities
        """
        probas = self._pyxit_classifier.predict_proba(X)
        best_index = np.argmax(probas, axis=1)
        return self._classes.take(best_index, axis=0).astype('int'), probas

    @staticmethod
    def build_from_pickle(model_path, tile_builder, logger, n_jobs=1):
        """Builds a PyxitClassifierAdapter object from a pickled model

        Parameters
        ----------
        model_path: string
            The path to which is stored the pickled model
        tile_builder: TileBuilder
            A tile builder object
        logger: Logger (optional, default: a SilentLogger object)
            A logger object
        n_jobs: int
            The number of jobs on which the classifiers should run

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
            classifier.n_jobs = n_jobs
            classifier.verbose = 10 if logger.level > Logger.DEBUG else 0
            classifier.base_estimator.n_jobs = n_jobs
            classifier.base_estimator.verbose = 10 if logger.level > logger.DEBUG else 0
        return PyxitClassifierAdapter(classifier, tile_builder, classes, logger=logger)
