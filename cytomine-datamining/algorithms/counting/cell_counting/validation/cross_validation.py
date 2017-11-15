# -*- coding: utf-8 -*-
import itertools
import time
from collections import defaultdict
from functools import partial

import numpy as np
from joblib import Parallel, delayed, logger as joblib_logger
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection import check_cv, GroupKFold, LeavePGroupsOut
from sklearn.model_selection._search import _check_param_grid, ParameterGrid
from sklearn.utils import indexable
from sklearn.utils.fixes import MaskedArray
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import check_is_fitted

from cell_counting.utils import open_scoremap, merge_dicts
from sldc import StandardOutputLogger, Logger

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


class GridSearchCV(BaseEstimator):
    def __init__(self, default_estimator, param_grid, cv, me, untrainable_param_grid=None,
                 scoring_rank='f1_score', refit=False, iid=True, n_jobs=1, pre_dispatch='2*n_jobs',
                 logger=StandardOutputLogger(Logger.INFO)):
        self.default_estimator = default_estimator
        self.param_grid = param_grid
        self.untrainable_param_grid = untrainable_param_grid
        self.cv = cv
        self.me = me
        self.scoring_rank = scoring_rank
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.logger = logger
        self.refit = refit
        self.iid = iid
        _check_param_grid(param_grid)
        _check_param_grid(untrainable_param_grid)

    def fit(self, X, y=None, groups=None):
        estimator = self.default_estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        # Regenerate parameter iterable for each fit
        candidate_params = ParameterGrid(self.param_grid)
        n_candidates = len(candidate_params)
        candidate_untrainable_params = ParameterGrid(self.untrainable_param_grid)
        untrainable_candidates = len(candidate_untrainable_params)
        self.logger.i("[CV] Fitting {} folds for each of {} candidates, totalling"
                      " {} fits".format(n_splits, n_candidates, n_candidates * n_splits))

        base_estimator = clone(self.default_estimator)
        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.logger.level * 20,
            pre_dispatch=pre_dispatch
        )(delayed(_fit_and_score)(clone(base_estimator), X, y, self.me,
                                  train, test, self.logger, parameters,
                                  candidate_untrainable_params,
                                  return_n_test_samples=True,
                                  return_times=True)
          for train, test in cv.split(X, y, groups)
          for parameters in candidate_params)

        out = np.vstack([o for o in out])
        test_accuracy = out[:, 0]
        test_precision = out[:, 1]
        test_recall = out[:, 2]
        test_f1 = out[:, 3]
        test_distance = out[:, 4]
        test_count = out[:, 5]
        test_count_pct = out[:, 6]
        test_raw_count = out[:, 7]
        test_raw_count_pct = out[:, 8]
        test_density = out[:, 9]
        test_raw_density = out[:, 10]
        test_sample_counts = out[:, 11]
        fit_time = out[:, 12]
        score_time = out[:, 13]

        results = dict()
        n_tot_candidates = n_candidates * untrainable_candidates
        tot_candidate_params = list(itertools.product(list(candidate_params), list(candidate_untrainable_params)))

        def _store(key_name, array, weights=None, splits=False, rank=False, error=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            array = np.array(array, dtype=np.float64).reshape(n_splits, n_tot_candidates).T
            if splits:
                for split_i in range(n_splits):
                    results["split%d_%s" % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                arr = array_means if error else -array_means
                results["rank_%s" % key_name] = np.asarray(rankdata(arr, method='min'), dtype=np.int32)

        # Computed the (weighted) mean and std for test scores alone
        # NOTE test_sample counts (weights) remain the same for all candidates
        if self.iid:
            test_sample_counts = np.array(test_sample_counts[::n_tot_candidates], dtype=np.int)
        else:
            test_sample_counts = None

        _store('accuracy_score', test_accuracy, splits=True, rank=True, weights=test_sample_counts)
        _store('precision_score', test_precision, splits=True, rank=True, weights=test_sample_counts)
        _store('recall_score', test_recall, splits=True, rank=True, weights=test_sample_counts)
        _store('f1_score', test_f1, splits=True, rank=True, weights=test_sample_counts)
        _store('distance_mae', test_distance, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('count_mae', test_count, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('count_pct_mae', test_count_pct, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('raw_count_mae', test_raw_count, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('raw_count_pct_mae', test_raw_count_pct, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('density_mae', test_density, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('raw_density_mae', test_raw_density, splits=True, rank=True, weights=test_sample_counts, error=True)
        _store('fit_time', fit_time)
        _store('score_time', score_time)
        results['rank_custom'] = np.asarray(rankdata((results['rank_f1_score'] + results['rank_count_pct_mae']) / 2,
                                                     method='min'), dtype=np.int32)

        best_index = np.flatnonzero(results['rank_custom'])[0]
        best_parameters = tot_candidate_params[best_index]

        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(MaskedArray, np.empty(n_tot_candidates, ), mask=True, dtype=object))
        for cand_i, params in enumerate(tot_candidate_params):
            params = merge_dicts(*params)
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)

        # Store a list of param dicts at the key 'params'
        results['params'] = tot_candidate_params

        self.cv_results_ = results
        self.best_index_ = best_index
        self.n_splits_ = n_splits

        if self.refit:
            bp = best_parameters[0]
            bp.update(best_parameters[1])
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(**bp)
            best_estimator.fit(X, y)
            self.best_estimator_ = best_estimator
        return self

    @property
    def best_params_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['params'][self.best_index_]

    @property
    def best_score_(self):
        check_is_fitted(self, 'cv_results_')
        return self.cv_results_['mean_test_score'][self.best_index_]


def _fit_and_score(estimator, X, y, me, train, test, logger,
                   parameters, candidate_untrainable_params,
                   return_n_test_samples=False,
                   return_times=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    logger : Logger
        The verbosity level.

    parameters : dict or None
        Parameters to be set on the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.

    test_score : float
        Score on test set.

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if parameters is None:
        msg = ''
    else:
        msg = '%s' % (', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
    logger.info("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    estimator.fit(X_train, y_train)
    fit_time = time.time() - start_time

    all_ret = []
    for untrainable_parameters in candidate_untrainable_params:
        me.reset()
        for x, y in itertools.izip(X_test, y_test):
            p = estimator.predict(np.array([x]))
            pp = estimator.postprocessing([p], **untrainable_parameters)
            me.compute([open_scoremap(y)], [pp], [p])
        metrics = me.all_metrics()
        score_time = time.time() - start_time - fit_time

        total_time = score_time + fit_time
        if parameters is None:
            msg1 = ''
        else:
            msg1 = '%s' % (', '.join('%s=%s' % (k, v) for k, v in untrainable_parameters.items()))
        end_msg = "%s %s, total=%s" % (msg1, msg, joblib_logger.short_format_time(total_time))
        logger.info("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

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
               metrics['raw_density']]

        if return_n_test_samples:
            ret.append(X_test.shape[0])
        if return_times:
            ret.extend([fit_time, score_time])

        all_ret.append(ret)

    return all_ret


def mk_tt_split(X, y, labels, test_labels):
    """
    Perform a train/test split based on labels.

    Parameters
    ----------
    X : array_like
        Input samples
    y : array_like
        Output samples
    labels : array_like
        Set of labels
    test_labels : array_like
        Set of test labels, that is, a subset of `labels`.

    Returns
    -------
    X_LS
    y_LS
    labels_LS
    X_TS
    y_TS
    labels_TS

    """
    test_set_labels = np.unique(test_labels)
    ts = np.in1d(labels, test_set_labels)
    ls = np.logical_not(ts)
    return (np.asarray(X[ls]), np.asarray(y[ls]), np.asarray(labels[ls]),
            np.asarray(X[ts]), np.asarray(y[ts]), np.asarray(labels[ts]))


def cv_strategy(parameters):
    if parameters.cv_mode == 'GKF':
        return GroupKFold(n_splits=parameters.cv_param)
    elif parameters.cv_mode == 'LPGO':
        return LeavePGroupsOut(n_groups=parameters.cv_param)
    else:
        raise ValueError("Unknown CV mode")


def mk_param_grid(param_dict, param_keys):
    ret = param_dict.copy()
    for k in param_dict.keys():
        if k not in param_keys:
            del ret[k]
    return ret

# def clone(estimator, safe=True):
#     """Constructs a new estimator with the same parameters.
#
#     Clone does a deep copy of the model in an estimator
#     without actually copying attached data. It yields a new estimator
#     with the same parameters that has not been fit on any data.
#
#     Parameters
#     ----------
#     estimator: estimator object, or list, tuple or set of objects
#         The estimator or group of estimators to be cloned
#
#     safe: boolean, optional
#         If safe is false, clone will fall back to a deepcopy on objects
#         that are not estimators.
#
#     """
#     estimator_type = type(estimator)
#     # XXX: not handling dictionaries
#     if estimator_type in (list, tuple, set, frozenset):
#         return estimator_type([clone(e, safe=safe) for e in estimator])
#     elif not hasattr(estimator, 'get_params'):
#         if not safe:
#             return copy.deepcopy(estimator)
#         else:
#             raise TypeError("Cannot clone object '%s' (type %s): "
#                             "it does not seem to be a scikit-learn estimator "
#                             "as it does not implement a 'get_params' cell_counting."
#                             % (repr(estimator), type(estimator)))
#     klass = estimator.__class__
#     new_object_params = estimator.get_params(deep=False)
#     for name, param in six.iteritems(new_object_params):
#         new_object_params[name] = clone(param, safe=False)
#     new_object = klass(**new_object_params)
#     params_set = new_object.get_params(deep=False)
#
#     # quick sanity check of the parameters of the clone
#     for name in new_object_params:
#         param1 = new_object_params[name]
#         param2 = params_set[name]
#         if param1 is param2:
#             # this should always happen
#             continue
#         if isinstance(param1, np.ndarray):
#             # For most ndarrays, we do not test for complete equality
#             if not isinstance(param2, type(param1)):
#                 equality_test = False
#             elif (param1.ndim > 0
#                   and param1.shape[0] > 0
#                   and isinstance(param2, np.ndarray)
#                   and param2.ndim > 0
#                   and param2.shape[0] > 0):
#                 equality_test = (
#                     param1.shape == param2.shape
#                     and param1.dtype == param2.dtype
#                     and (_first_and_last_element(param1) ==
#                          _first_and_last_element(param2))
#                 )
#             else:
#                 equality_test = np.all(param1 == param2)
#         elif sparse.issparse(param1):
#             # For sparse matrices equality doesn't work
#             if not sparse.issparse(param2):
#                 equality_test = False
#             elif param1.size == 0 or param2.size == 0:
#                 equality_test = (
#                     param1.__class__ == param2.__class__
#                     and param1.size == 0
#                     and param2.size == 0
#                 )
#             else:
#                 equality_test = (
#                     param1.__class__ == param2.__class__
#                     and (_first_and_last_element(param1) ==
#                          _first_and_last_element(param2))
#                     and param1.nnz == param2.nnz
#                     and param1.shape == param2.shape
#                 )
#         else:
#             # fall back on standard equality
#             equality_test = param1 == param2
#         if equality_test:
#             pass
#         else:
#             raise RuntimeError('Cannot clone object %s, as the constructor '
#                                'does not seem to set parameter %s' %
#                                (estimator, name))
#
#     return new_object
