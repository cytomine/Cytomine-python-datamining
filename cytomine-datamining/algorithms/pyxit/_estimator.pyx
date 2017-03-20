# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython

import numpy as np
cimport numpy as np

import os
import cPickle


def inplace_csr_column_scale_max(X, np.ndarray[np.float32_t, ndim=1] maxs=None):
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    cdef np.ndarray[np.float32_t, ndim=1] X_data = X.data
    cdef np.ndarray[int, ndim=1] X_indices = X.indices
    cdef np.ndarray[int, ndim=1] X_indptr = X.indptr

    cdef unsigned int i
    cdef unsigned int j
    cdef unsigned int ind

    if maxs is None:
        maxs = np.zeros(n_features, dtype=np.float32)

        for i in range(n_samples):
            for j in range(X_indptr[i], X_indptr[i + 1]):
                ind = X_indices[j]

                if X_data[j] > maxs[ind]:
                    maxs[ind] = X_data[j]

    for i in range(n_samples):
        for j in range(X_indptr[i], X_indptr[i + 1]):
            ind = X_indices[j]

            if maxs[ind] > 0.0:
                X_data[j] /= maxs[ind]

    return X, maxs
