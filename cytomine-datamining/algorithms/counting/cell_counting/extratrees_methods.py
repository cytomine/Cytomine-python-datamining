# -*- coding: utf-8 -*-

import pickle
import types
from functools import partial

import numpy as np
from sklearn.ensemble.forest import (
    ForestClassifier,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)

from cell_counting.base_method import BaseMethod
from cell_counting.subwindows import mk_subwindows, half_size, all_subwindows_generator, subwindow_box
from sldc import Logger, StandardOutputLogger
from cell_counting.utils import open_image_with_mask

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


class CellCountRandomizedTrees(BaseMethod):
    def __init__(self, build_fn=None, logger=StandardOutputLogger(Logger.INFO), **sk_params):
        super(CellCountRandomizedTrees, self).__init__(build_fn, logger, **sk_params)
        self.__forest = None

    def fit(self, X, y, _X=None, _y=None):
        if self.build_fn is None:
            self.__forest = self.build_rt(**self.filter_sk_params(self.build_rt))
        elif not isinstance(self.build_fn, types.FunctionType) and not isinstance(self.build_fn, types.MethodType):
            self.__forest = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.__forest = self.build_fn(**self.filter_sk_params(self.build_fn))

        X, y = np.asarray(X), np.asarray(y)
        if _X is None and _y is None:
            _X, _y = self.extract_subwindows(X, y)

        self.logger.i('[FIT] Start fitting from {} images, {} subwindows'.format(X.shape[0], _X.shape[0]))
        self.__forest.fit(_X, _y)

        if isinstance(self.__forest, ForestClassifier):
            self.foreground_class_ = np.where(self.__forest.classes_ == 1.)

        return self

    def extract_subwindows(self, X, y, labels=None):
        X, y = np.asarray(X), np.asarray(y)

        self.logger.i("[EXTRACT SUBWINDOWS] Start extracting subwindows from {} images".format(X.shape[0]))
        _X, _y = mk_subwindows(X, y, labels, **self.filter_sk_params(mk_subwindows))

        self.logger.i("[EXTRACT SUBWINDOWS] _X size: ({} samples / {} features)".format(_X.shape[0], _X.shape[1]))
        self.logger.i("[EXTRACT SUBWINDOWS] _y size: ({} samples)".format(_y.shape[0]))
        return _X, _y

    def _predict_clf_helper(self, _x, foreground_class):
        foreground_class = np.asarray(foreground_class).squeeze()
        return self.__forest.predict_proba(_x)[:, foreground_class]

    def predict(self, X):
        self.logger.i("[PREDICT]")
        ret_lst = []
        for x in X:
            if hasattr(self, 'foreground_class_'):
                predict_method = partial(self._predict_clf_helper, foreground_class=self.foreground_class_)
            else:
                predict_method = self.__forest.predict

            window_input_size_half = half_size(self.sk_params['sw_input_size'])
            window_output_size_half = half_size(self.sk_params['sw_output_size'])
            image, mask = open_image_with_mask(x, padding=window_input_size_half)
            y = np.zeros_like(mask, dtype=np.float16)
            count = np.zeros_like(mask, dtype=np.uint16)

            asg = all_subwindows_generator(image, mask, batch_size=image.shape[0] * image.shape[1],
                                           **self.filter_sk_params(all_subwindows_generator))
            for sws, coords in asg:
                predictions = predict_method(sws)
                for prediction, coord in zip(predictions, coords):
                    top, right, bottom, left = subwindow_box(self.sk_params['sw_output_size'],
                                                             window_output_size_half, coord)
                    y[slice(top, bottom), slice(left, right)] += prediction.reshape(self.sk_params['sw_output_size'])
                    count[slice(top, bottom), slice(left, right)] += 1

            y[count > 1] = y[count > 1] / count[count > 1]

            ret_lst.append(y[window_input_size_half[0]: -window_input_size_half[0],
                           window_input_size_half[1]: -window_input_size_half[1]])

        return np.squeeze(ret_lst)

    @property
    def forest(self):
        return self.__forest

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def build_rt(forest_method, forest_n_estimators, forest_min_samples_split, forest_max_features, n_jobs):
        if 'ET' in forest_method:
            if 'clf' in forest_method:
                MLAlgo = partial(ExtraTreesClassifier, class_weight='balanced')
            else:
                MLAlgo = ExtraTreesRegressor
        else:
            if 'clf' in forest_method:
                MLAlgo = partial(RandomForestClassifier, class_weight='balanced')
            else:
                MLAlgo = RandomForestRegressor

        return MLAlgo(
            n_estimators=forest_n_estimators,
            min_samples_split=forest_min_samples_split,
            max_features=forest_max_features,
            n_jobs=n_jobs)
