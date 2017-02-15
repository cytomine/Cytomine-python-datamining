# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
import tempfile

from cytomine_sldc import TileCache
from sklearn.utils import check_random_state

from sldc import PolygonClassifier, TileExtractionException, Loggable, SilentLogger, Logger

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def _crops_extract(image, polygons, tile_cache):
    """Extract and cache the crops for the given polygons

    Parameters
    ----------
    image: Image
        The image from which the crops must be taken
    polygons: iterable (subtype: shapely.geometry.Polygon)
        The polygons of which the crops must be feteched
    tile_cache: TileCache
        The tile cache object

    Returns
    -------
    paths: iterable (subtype: string)
        The path of the caching file for the given image
    extracted: iterable (subtype: int)
        The indexes of the polygons successfully extracted
    errors: iterable (subtype: string)
        The error message explaining why some tiles couldn't be fetched. One error per non-fetched polygon.
    """
    errors = list()
    paths = list()
    extracted = list()
    for i, polygon in enumerate(polygons):
        try:
            paths.append(tile_cache.polygon_fetch_and_cache(image, polygon))
            extracted.append(i)
        except TileExtractionException as e:
            errors.append(e.message)
    return paths, extracted, errors


class PyxitClassifierAdapter(PolygonClassifier, Loggable):

    def __init__(self, pyxit_classifier, tile_builder, classes, logger=SilentLogger(), working_path=None):
        """Constructor for PyxitClassifierAdapter objects

        Parameters
        ----------
        pyxit_classifier: PyxitClassifier
            The pyxit classifier objects
        tile_builder: TileBuilder
            A tile builder for fetching crop tiles
        classes: ndarray
            An array containing the classes labels
        logger: Logger
            A logger object
        working_path: str (optional, default: temp folder)
            Folder in which will be saved temporary crop images
        """
        PolygonClassifier.__init__(self)
        Loggable.__init__(self, logger)
        self._pyxit_classifier = pyxit_classifier
        self._working_path = working_path if working_path is not None else os.path.join(tempfile.gettempdir(), "sldc")
        self._tile_cache = TileCache(tile_builder, self._working_path)
        self._classes = classes

    def predict(self, image, polygon):
        return self.predict(image, [polygon])[0]

    def predict_batch(self, image, polygons):
        # extract crops of images to classify
        # suppose that no error will occur during the transfer
        self.logger.i("Extract crops...")
        extracted, _, _ = _crops_extract(image, polygons, self._tile_cache)

        # actually predicts the class for each crop
        self.logger.i("Predict classes for crops...")
        return self._predict(np.array(extracted))

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
        return self._classes.take(best_index, axis=0).astype('int'), np.max(probas, axis=1)

    @staticmethod
    def build_from_pickle(model_path, tile_builder, logger, random_state=None, n_jobs=1, working_path=None):
        """Builds a PyxitClassifierAdapter object from a pickled model

        Parameters
        ----------
        model_path: string
            The path to which is stored the pickled model
        tile_builder: TileBuilder
            A tile builder object
        logger: Logger (optional, default: a SilentLogger object)
            A logger object
        random_state: Random (optional, default: None)
            Random number generator to be passed to the pyxit classifier
        n_jobs: int
            The number of jobs on which the classifiers should run
        working_path: str
            Path where the PyxitClassifierAdapter should store temporary crop images


        Returns
        -------
        adapter: PyxitClassifierAdapter
            The built classifier adapter
        """
        random_state = check_random_state(random_state)
        with open(model_path, "rb") as model_file:
            _ = pickle.load(model_file)
            classes = pickle.load(model_file)
            classifier = pickle.load(model_file)
            classifier.n_jobs = n_jobs
            classifier.verbose = 10 if logger.level >= Logger.DEBUG else 0
            classifier.base_estimator.n_jobs = n_jobs
            classifier.base_estimator.verbose = 10 if logger.level >= logger.DEBUG else 0
            classifier.random_state = random_state
            classifier.classes_ = classes
        return PyxitClassifierAdapter(classifier, tile_builder, classes, logger=logger, working_path=working_path)
