# -*- coding: utf-8 -*-
import numpy as np
from cytomine.models import AnnotationCollection
from pyxit.estimator import PyxitClassifier, _get_output_from_directory
from sklearn.svm import LinearSVC

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
