# -*- coding: utf-8 -*-
import copy
import inspect
from collections import defaultdict
from functools import partial

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid
from sklearn.utils.fixes import MaskedArray

from cell_counting.postprocessing import non_maximum_suppression
from cell_counting.utils import open_scoremap
from sldc import StandardOutputLogger, Logger

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


class BaseMethod(object):
    def __init__(self, build_fn=None, logger=StandardOutputLogger(Logger.INFO), **sk_params):
        self.build_fn = build_fn
        self.sk_params = sk_params
        self.logger = logger

    def get_params(self, **params):
        """Gets parameters for this estimator.

        # Arguments
            **params: ignored (exists for API compatiblity).

        # Returns
            Dictionary of parameter names mapped to their values.
        """
        res = copy.deepcopy(self.sk_params)
        res.update({'build_fn': self.build_fn})
        return res

    def set_params(self, **params):
        """Sets the parameters of this estimator.

        # Arguments
            **params: Dictionary of parameter names mapped to their values.

        # Returns
            self
        """
        self.sk_params.update(params)
        return self

    def filter_sk_params(self, fn, override=None, exceptions=[]):
        """Filters `sk_params` and return those in `fn`'s arguments.

        # Arguments
            fn : arbitrary function
            override: dictionary, values to override sk_params

        # Returns
            res : dictionary dictionary containing variables
                in both sk_params and fn's arguments.
        """
        override = override or {}
        res = {}
        fn_args = inspect.getargspec(fn)[0]
        for name, value in self.sk_params.items():
            if name in fn_args and name not in exceptions:
                res.update({name: value})
        res.update(override)
        return res

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def postprocessing(self, X, **post_params):
        return np.squeeze([non_maximum_suppression(x, **post_params) for x in X])

    def score(self, X_test, y_test, me, untrainable_param_grid):
        candidate_untrainable_params = ParameterGrid(untrainable_param_grid)
        n_untrainable_candidates = len(candidate_untrainable_params)
        all_ret = []
        for untrainable_parameters in candidate_untrainable_params:
            me.reset()
            for x, y in np.itertools.izip(X_test, y_test):
                p = self.predict(np.array([x]))
                pp = self.postprocessing([p], **untrainable_parameters)
                me.compute([open_scoremap(y)], [pp], [p])
            metrics = me.all_metrics()

            ret = [metrics['accuracy'],
                   metrics['precision'],
                   metrics['recall'],
                   metrics['f1'],
                   metrics['distance'],
                   metrics['count'],
                   metrics['count_pct'],
                   metrics['raw_count'],
                   metrics['raw_count_pct'],
                   metrics['density'],
                   metrics['raw_density'],
                   X_test.shape[0]]

            all_ret.append(ret)

        all_ret = np.asarray(all_ret)
        results = dict()

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(1, n_untrainable_candidates).T

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(rankdata(-array_means, method='min'), dtype=np.int32)

        test_sample_counts = all_ret[:, 11]
        test_sample_counts = np.array(test_sample_counts[::n_untrainable_candidates], dtype=np.int)
        _store('accuracy_score', all_ret[:, 0], rank=True, weights=test_sample_counts)
        _store('precision_score', all_ret[:, 1], rank=True, weights=test_sample_counts)
        _store('recall_score', all_ret[:, 2], rank=True, weights=test_sample_counts)
        _store('f1_score', all_ret[:, 3], rank=True, weights=test_sample_counts)
        _store('distance_mae', all_ret[:, 4], rank=True, weights=test_sample_counts)
        _store('count_mae', all_ret[:, 5], rank=True, weights=test_sample_counts)
        _store('count_pct_mae', all_ret[:, 6], rank=True, weights=test_sample_counts)
        _store('raw_count_mae', all_ret[:, 7], rank=True, weights=test_sample_counts)
        _store('raw_count_pct_mae', all_ret[:, 8], rank=True, weights=test_sample_counts)
        _store('density_mae', all_ret[:, 9], rank=True, weights=test_sample_counts)
        _store('raw_density_mae', all_ret[:, 10], rank=True, weights=test_sample_counts)

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray, np.empty(n_untrainable_candidates, ), mask=True, dtype=object))
        for cand_i, params in enumerate(list(candidate_untrainable_params)):
            # params = merge_dicts(*params)
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        return results

    def save(self, filename):
        raise NotImplementedError()
