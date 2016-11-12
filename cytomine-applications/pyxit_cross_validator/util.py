# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

__author__          = "Mormont Romain <r.mormont@ulg.ac.be>"
__contributors__    = []
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

import os

import shutil

from cytomine.models import Annotation
from sklearn.metrics import precision_score, accuracy_score, recall_score


__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels:
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2list(l, conv=int):
    return [conv(v) for v in l.split(",")] if len(l) > 0 else []


def default(str, default):
    return str if len(str) > 0 else default


def create_dir(path, clean=False):
    if not os.path.exists(path):
        print "Creating annotation directory: %s" % path
        os.makedirs(path)
    elif clean:
        print "Cleaning annotation directory: %s" % path
        shutil.rmtree(path)
        os.makedirs(path)


def copy_content(src, dst):
    files = [f for f in os.listdir(src) if not os.path.isdir(os.path.join(src, f))]
    for f in files:
        shutil.copy(os.path.join(src, f), dst)


class CropTypeEnum(object):
    CROP = 1
    ALPHA_CROP = 2

    @staticmethod
    def enum2crop(enum):
        if enum == CropTypeEnum.CROP:
            return Annotation.get_annotation_crop_url
        elif enum == CropTypeEnum.ALPHA_CROP:
            return Annotation.get_annotation_alpha_crop_url
        else:
            raise ValueError("Invalid enum field : {}".format(enum))


def accuracy_scoring(pyxit, X, y):
    """A scikit-learn compatible accuracy scoring function"""
    return accuracy_score(y, pyxit.predict(X))


def recall_scoring(pyxit, X, y):
    """A scikit-learn compatible recall scoring function"""
    return recall_score(y, pyxit.predict(X))


def precision_scoring(pyxit, X, y):
    """A scikit-learn compatible precision scoring function"""
    return precision_score(y, pyxit.predict(X))


def get_greater(value, lst):
    """Get all the value from the list that are greater than the given value"""
    return [v for v in lst if value < v]


def mk_window_size_tuples(min_sizes, max_sizes):
    """Generate a list of all possible pairs (min_window_size, max_window_size) from the list of
    minimum and maximum windows sizes"""
    min_sizes = sorted(min_sizes)
    max_sizes = sorted(max_sizes)
    tuples = list()
    for min_size in min_sizes:
        valid_sizes = get_greater(min_size, max_sizes)
        if len(valid_sizes) > 0:
            tuples += [(min_size, max_size) for max_size in valid_sizes]
    return tuples


class Logger(object):
    """A class for logging based on verbosity"""
    def __init__(self, verbose):
        self._verbose = verbose

    def log(self, content):
        if self._verbose:
            print(content)