# -*- coding: utf-8 -*-
import itertools
import os

import numpy as np
from sklearn.neighbors import KDTree

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def _perform_divide(numerator, denominator):
    if denominator == 0.:
        return 0
    else:
        return float(numerator) / float(denominator)


def _perform_mean(lst, dtype=np.float):
    if len(lst) == 0:
        return 0.
    else:
        return np.mean(lst, dtype=dtype)


class ConfusionMatrix(object):
    """
    A confusion matrix.
    """

    def __init__(self):
        self.cm = np.zeros((2, 2))

    def incrementTP(self, value=1):
        self.cm[0, 0] += value

    def incrementFP(self, value=1):
        self.cm[1, 0] += value

    def incrementFN(self, value=1):
        self.cm[0, 1] += value

    def incrementTN(self, value=1):
        self.cm[1, 1] += value

    def increment(self, other):
        self.incrementTP(other.tp)
        self.incrementFP(other.fp)
        self.incrementFN(other.fn)
        self.incrementTN(other.tn)

    @property
    def tp(self):
        return self.cm[0, 0]

    @property
    def fp(self):
        return self.cm[1, 0]

    @property
    def fn(self):
        return self.cm[0, 1]

    @property
    def tn(self):
        return self.cm[1, 1]

    @property
    def confusion_matrix(self):
        return self.cm

    def precision(self):
        """
        Returns
        -------
        precision: float
            A score equals to TP/(TP+FP)
        """
        return _perform_divide(self.tp, self.tp + self.fp)

    def recall(self):
        """
        Returns
        -------
        recall: float
            A score equals to TP/(TP+FN)
        """
        return _perform_divide(self.tp, self.tp + self.fn)

    def f1(self):
        """
        Returns
        -------
        f1: float
            A score equals to (2 * precision * recall)/(precision + recall)
        """
        p = self.precision()
        r = self.recall()
        return _perform_divide(2 * p * r, p + r)

    def accuracy(self):
        """
        Returns
        -------
        accuracy: float
            A score equals to (TP+TN)/(TN+TN+FP+FN)
        """
        return _perform_divide(self.tp + self.tn, np.sum(self.cm))

    def __str__(self):
        s = "    TP: {} | FN: {}".format(self.tp, self.fn)
        s += os.linesep + "    FP: {} | TN: {}".format(self.fp, self.tn)
        return s


class MetricsEvaluator(object):
    """
    Helper to evaluate metrics and store results.
    """

    def __init__(self, epsilon, raw_factor=1.):
        """
        Initialize the metrics.

        Parameters
        ----------
        epsilon: int
            The maximum radius (in pixels) around a groundtruth to associate it 
            with a detection.
        """
        self._epsilon = epsilon
        self._raw_factor = raw_factor
        self.reset()

    def reset(self):
        self.cm = ConfusionMatrix()
        self.distance_errors = list()
        self.count_errors = list()
        self.raw_count_errors = list()
        self.density_errors = list()
        self.raw_density_errors = list()
        self.count = 0.

    @property
    def confusion_matrix(self):
        return self.cm.confusion_matrix

    @property
    def accuracy_score(self):
        return self.cm.accuracy()

    @property
    def precision_score(self):
        return self.cm.precision()

    @property
    def recall_score(self):
        return self.cm.recall()

    @property
    def f1_score(self):
        return self.cm.f1()

    @property
    def distance_MAE(self):
        return _perform_mean(self.distance_errors, dtype=np.float)

    @property
    def count_MAE(self):
        return _perform_mean(self.count_errors, dtype=np.float)

    @property
    def pct_count_MAE(self):
        return np.sum(self.count_errors) / float(self.count) * 100.

    @property
    def raw_count_MAE(self):
        return _perform_mean(self.raw_count_errors, dtype=np.float)

    @property
    def pct_raw_count_MAE(self):
        return np.sum(self.raw_count_errors) / float(self.count) * 100.

    @property
    def density_MAE(self):
        return _perform_mean(self.density_errors, dtype=np.float)

    @property
    def raw_density_MAE(self):
        return _perform_mean(self.raw_density_errors, dtype=np.float)

    def all_metrics(self):
        metrics = dict()
        metrics['accuracy'] = self.accuracy_score
        metrics['precision'] = self.precision_score
        metrics['recall'] = self.recall_score
        metrics['f1'] = self.f1_score
        metrics['distance'] = self.distance_MAE
        metrics['count'] = self.count_MAE
        metrics['count_pct'] = self.pct_count_MAE
        metrics['raw_count'] = self.raw_count_MAE
        metrics['raw_count_pct'] = self.pct_raw_count_MAE
        metrics['density'] = self.density_MAE
        metrics['raw_density'] = self.raw_density_MAE
        return metrics

    def summary(self):
        return pprint_scores(self.all_metrics(), self._epsilon, self.cm)

    def compute(self, images_true, images_pred, images_score):
        """
        Update the metrics accumulators according to results of the provided
        samples.

        Parameters
        ----------
        images_true: list of array-like of shape (width, height)
            The true scoremaps.
        images_pred: list of array-like of shape (width, height)
            The predicted binary masks.
        images_score: list of array-like of shape (width, height)
            The predicted scoremaps.
        """
        if not (len(images_true) == len(images_pred) == len(images_score)):
            raise ValueError('Dimensions must correspond.')

        for image_true, image_pred, image_score in itertools.izip(images_true, images_pred, images_score):
            groundtruths = np.argwhere(np.asarray(image_true) > 0)
            predictions = np.argwhere(np.asarray(image_pred) > 0)

            n_groundtruths = float(groundtruths.shape[0])
            n_predictions = float(predictions.shape[0])
            width = float(image_true.shape[0])
            height = float(image_true.shape[1])
            raw_count = np.sum(image_score / np.float(self._raw_factor), dtype=np.float64)

            # Sort detections by score in descending order such that the first detection
            # linked to a ground truths is the more confident one
            scores = image_score[tuple(predictions.T)]
            predictions = predictions[np.argsort(scores, kind='quicksort')]

            n_tp = 0
            if n_predictions > 0:
                kdtree = KDTree(predictions, metric='euclidean')
                pred_idxss, distancess = kdtree.query_radius(groundtruths, self._epsilon, return_distance=True)
                free = np.array([True] * predictions.shape[0])
                for pred_idxs, distances in itertools.izip(pred_idxss, distancess):
                    pred_idxs = pred_idxs[free]
                    if len(pred_idxs) > 0 and n_tp <= int(n_predictions):
                        pred_idx = pred_idxs[np.argmax(pred_idxs)]
                        free[pred_idx] = False
                        self.distance_errors.append(distances[np.argmax(pred_idxs)])
                        n_tp += 1

            self.cm.incrementTP(n_tp)
            self.cm.incrementFP(int(n_predictions - n_tp))
            self.cm.incrementFN(int(n_groundtruths - n_tp))

            self.count_errors.append(np.abs(n_groundtruths - n_predictions))
            self.raw_count_errors.append(np.abs(n_groundtruths - raw_count))

            self.density_errors.append(np.abs((n_groundtruths / (width * height)) - (n_predictions / (width * height))))
            self.raw_density_errors.append(np.abs((n_groundtruths / (width * height)) - (raw_count / (width * height))))
            self.count += n_groundtruths


def pprint_scores(metrics, epsilon, cm=None):
    s = '=' * 80
    s += os.linesep + 'Scores obtained with epsilon fixed to {}'.format(epsilon)
    if cm is not None:
        s += os.linesep + 'Confusion matrix:'
        s += os.linesep + cm.__str__()
    s += os.linesep + 'Accuracy:  {}'.format(metrics['accuracy'])
    s += os.linesep + 'Precision: {}'.format(metrics['precision'])
    s += os.linesep + 'Recall:    {}'.format(metrics['recall'])
    s += os.linesep + 'F1-Score:  {}'.format(metrics['f1'])
    s += os.linesep + 'Distance MAE:    {}'.format(metrics['distance'])
    s += os.linesep + 'Count MAE:       {} ({}%)'.format(metrics['count'], metrics['count_pct'])
    s += os.linesep + 'Raw count MAE:   {} ({}%)'.format(metrics['raw_count'], metrics['raw_count_pct'])
    s += os.linesep + 'Density MAE:     {}'.format(metrics['density'])
    s += os.linesep + 'Raw density MAE: {}'.format(metrics['raw_density'])
    s += os.linesep + '=' * 80
    return s


if __name__ == '__main__':
    pass
    # from jobs.region_of_interest_ import rois_from_dataset
    # from cell_counting.utils import open_scoremap
    #
    # X, y, labels = list(), list(), list()
    # rois = rois_from_dataset('BMGRAZ',
    #                          working_path='/Users/ulysse/Documents/Programming/TFE/tmp/',
    #                          force_download=False)
    # for r in rois:
    #     X.append(r.image_filename)
    #     y.append(open_scoremap(r.groundtruth_filename))
    #     labels.append(r.roi_id)
    #
    # me = MetricsEvaluator(epsilon=10)
    # y = np.array(y)
    # me.compute(y, y, y)
    # print me.summary()
