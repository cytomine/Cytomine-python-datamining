# -*- coding: utf-8 -*-
from cytomine.models import AnnotationCollection
from pyxit.estimator import PyxitClassifier, _get_output_from_directory

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class AnnotationCollectionAdapter(object):
    def __init__(self, collection):
        if isinstance(collection, AnnotationCollection):
            self._annotations = collection.data()
        else:
            self._annotations = collection

    def data(self):
        return self._annotations

    def __iter__(self):
        for a in self._annotations:
            yield a


class PyxitClassifierAdapter(PyxitClassifier):

    def __init__(self, base_estimator,
                       n_subwindows=10,
                       min_size=0.5,
                       max_size=1.0,
                       target_width=16,
                       target_height=16,
                       n_jobs=1,
                       interpolation=2,
                       transpose=False,
                       colorspace=2,
                       fixed_size=False,
                       random_state=None,
                       verbose=0,
                       get_output = _get_output_from_directory,
                       window_sizes=None):
        PyxitClassifier.__init__(self, base_estimator, n_subwindows=n_subwindows, min_size=min_size, max_size=max_size,
                                 target_width=target_width, target_height=target_height, n_jobs=n_jobs,
                                 interpolation=interpolation, transpose=transpose, colorspace=colorspace,
                                 fixed_size=fixed_size, random_state=random_state, verbose=verbose,
                                 get_output=get_output)
        if window_sizes is not None:
            self.min_size = window_sizes[0]
            self.max_size = window_sizes[1]

    def get_params(self, deep=True):
        parameters = self.base_estimator.get_params() if deep else dict()
        parameters["base_estimator"] = self.base_estimator
        parameters["n_subwindows"] = self.n_subwindows
        parameters["min_size"] = self.min_size
        parameters["max_size"] = self.max_size
        parameters["window_sizes"] = self.min_size, self.max_size
        parameters["target_width"] = self.target_width
        parameters["target_height"] = self.target_height
        parameters["n_jobs"] = self.n_jobs
        parameters["interpolation"] = self.interpolation
        parameters["transpose"] = self.transpose
        parameters["colorspace"] = self.colorspace
        parameters["fixed_size"] = self.fixed_size
        parameters["verbose"] = self.verbose
        parameters["get_output"] = self.get_output
        return parameters

    def set_params(self, **params):
        wsize_removed = dict(params)
        super(PyxitClassifierAdapter, self).set_params(**wsize_removed)
        if "window_sizes" in params and params["window_sizes"] is not None:
            self.min_size, self.max_size = params["window_sizes"]
        return self


