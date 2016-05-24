# -*- coding: utf-8 -*-
import pickle
import numpy as np
import time
from guppy import hpy
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from sldc import PolygonClassifier, TileExtractionException, Loggable, SilentLogger, Logger, batch_split

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def _parallel_extract(image, polygons, tile_cache):
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

    def __init__(self, pyxit_classifier, tile_cache, classes, svm=None, logger=SilentLogger(), n_jobs=1):
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
        svm: LinearSVC (optional, default: None)
            The SVC classifier to apply the svm layer above Pyxit
        n_jobs: int (optional, default: 1)
            The number of jobs for fetching the crops
        """
        PolygonClassifier.__init__(self)
        Loggable.__init__(self, logger)
        self._pyxit_classifier = pyxit_classifier
        self._tile_cache = tile_cache
        self._classes = classes
        self._svm = svm
        self._pool = Parallel(n_jobs=n_jobs)
        self._heapy = hpy()

    def predict_batch(self, image, polygons):
        # Pyxit classifier takes images from the filesystem
        # So store the crops into files before passing the paths to the classifier
        hp_before_load = self._heapy.heap()
        start_load = time.time()
        poly_batches = batch_split(self._pool.n_jobs, polygons)
        fetched = self._pool(delayed(_parallel_extract)(image, batch, self._tile_cache) for batch in poly_batches)
        end_load = time.time()
        hp_end_load = self._heapy.heap()
        print "Fetching tiles from server ({} sec):".format(end_load - start_load)
        print hp_end_load - hp_before_load

        count_missing = 0
        paths = list()
        extracted = list()
        current_batch_size = 0
        for i, (b_paths, b_extracted, b_errors) in enumerate(fetched):
            paths += b_paths
            extracted += (np.array(b_extracted) + current_batch_size).tolist()
            current_batch_size += len(poly_batches[i])
            count_missing += len(b_errors)
            # print errors
            for error in b_errors:
                self.logger.e("PyxitClassifierAdapter: skip polygon because tile cannot be extracted.\n" +
                              "PyxitClassifierAdapter: error : {}".format(error))

        self.logger.w("PyxitClassifierAdapter: {} crop(s) missed".format(count_missing))

        # merge predictions with missed tiles
        hp_before_predict = self._heapy.heap()
        start_predict = time.time()
        predictions, probas = self._predict(np.array(paths))
        end_predict = time.time()
        hp_after_predict = self._heapy.heap()
        print "Predictiting ({}):".format(end_predict - start_predict)
        print hp_after_predict - hp_before_predict

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
        if self._svm is None:
            self.logger.i("PyxitClassifierAdapter: predict without svm.")
            probas = self._pyxit_classifier.predict_proba(X)
            best_index = np.argmax(probas, axis=1)
            return self._classes.take(best_index, axis=0).astype('int'), probas
        else:
            self.logger.i("PyxitClassifierAdapter: predict with svm.")
            Xt = self._pyxit_classifier.transform(X)
            self.logger.i("PyxitClassifierAdapter: predict with svm, features generated.")
            if hasattr(self._svm, "predict_proba"):
                probas = self._svm.predict_proba(Xt)
                best_index = np.argmax(probas, axis=1)
                return self._classes.take(best_index, axis=0).astype('int'), probas
            else:
                return self._svm.predict(Xt), np.ones((X.shape[0],))

    @staticmethod
    def build_from_pickle(model_path, tile_builder, logger, random_state=None, n_jobs=1):
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

        Returns
        -------
        adapter: PyxitClassifierAdapter
            The built classifier adapter

        Notes
        -----
        The first object pickled in the file in the path 'model_path' is an array
        containing the classes, and the second is the PyxitClassifier object
        """
        random_state = check_random_state(random_state)
        with open(model_path, "rb") as model_file:
            type = pickle.load(model_file)
            classes = pickle.load(model_file)
            classifier = pickle.load(model_file)
            classifier.n_jobs = n_jobs
            classifier.verbose = 10 if logger.level >= Logger.DEBUG else 0
            classifier.base_estimator.n_jobs = n_jobs
            classifier.base_estimator.verbose = 10 if logger.level >= logger.DEBUG else 0
            classifier.interpolation = 1  # nearest interpolation
            classifier.random_state = random_state
            svm = pickle.load(model_file) if type == "svm" else None
        return PyxitClassifierAdapter(classifier, tile_builder, classes, svm=svm, logger=logger, n_jobs=min(10, n_jobs))
