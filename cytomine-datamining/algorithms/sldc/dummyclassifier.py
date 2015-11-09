# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'

import numpy as np
import copy_reg
import types

class DummyClassifier(object):

    def __init__(self, val):
        self.val = val

    def predict(self, img_stream):
        return np.ones(len(img_stream), dtype=np.int) * self.val


def piclking_reduction(m):
    """Adds the capacity to pickle method of objects"""
    return (getattr, (m.__self__, m.__func__.__name__))



if __name__ == "__main__":

    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    copy_reg.pickle(types.MethodType, piclking_reduction)

    with open("cell_classif", "wb") as f:
        pickle.dump(DummyClassifier(15054765), f)
    with open("arch_pattern_classif", "wb") as f:
        pickle.dump(DummyClassifier(15054705), f)
