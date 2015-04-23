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

__author__          = "Gilles Louppe"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>", "Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import numpy as np
import math
import sys

try:
    import Image
except:
    from PIL import Image

from scipy.sparse import csr_matrix
from scipy.stats.mstats import mode

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn.utils import check_random_state

from _estimator import leaf_transform
from _estimator import inplace_csr_column_scale_max

MAX_INT = np.iinfo(np.int32).max

INTERPOLATION_NEAREST = 1
INTERPOLATION_BILINEAR = 2
INTERPOLATION_CUBIC = 3
INTERPOLATION_ANTIALIAS = 4

COLORSPACE_RGB = 0
COLORSPACE_TRGB = 1
COLORSPACE_HSV = 2
COLORSPACE_GRAY = 3



def _raw_to_rgb(raw):
    return raw.flatten()


def _raw_to_trgb(raw):
    assert raw.shape[1] == 3

    mean = np.atleast_1d(np.mean(raw, axis=0))
    std = np.atleast_1d(np.std(raw, axis=0))

    trgb = np.zeros(raw.shape)

    for i, s in enumerate(std):
        if np.abs(s) > 10E-9: # Do to divide by zero
            trgb[:, i] = (raw[:, i] - mean[i]) / s

    return trgb.flatten()


def _raw_to_hsv(raw):
    assert raw.shape[1] == 3

    # Min/Max/Diff
    dim = raw.shape[0]
    fmin = np.min(raw, axis=1)
    fmax = np.max(raw, axis=1)
    diff = fmax - fmin

    # Value
    value = np.asarray(fmax, dtype=np.float32)

    # Sat
    sat = np.zeros(dim, dtype=np.float32)
    mask = fmax > 0.0
    sat[mask] = diff[mask] / fmax[mask]

    # Hue
    hue = np.zeros(dim, dtype=np.float32)
    mask = sat > 0.0

    mask_r = mask & (raw[:, 0] == fmax)
    mask_g = mask & (raw[:, 1] == fmax)
    mask_b = mask & (raw[:, 2] == fmax)

    hue[mask_r] = (raw[mask_r, 1] - raw[mask_r, 2]) / diff[mask_r]
    hue[mask_g] = (raw[mask_g, 2] - raw[mask_g, 0]) / diff[mask_g]
    hue[mask_g] += 2.0
    hue[mask_b] = (raw[mask_b, 0] - raw[mask_b, 1]) / diff[mask_b]
    hue[mask_b] += 4.0

    hue *= 60.0
    hue[hue < 0.0] += 360.0
    hue[hue > 360.0] -= 360.

    return np.hstack((hue[:, np.newaxis], sat[:, np.newaxis], value[:, np.newaxis])).flatten()


def _raw_to_gray(raw):
    #print "raw shape: %d" %raw.shape[1]
    return 1.0 * np.sum(raw, axis=1) / raw.shape[1]


#Random subwindows extraction (Maree et al., 2014). It extracts subwindows of random sizes at random locations in images (fully contains in the image)
def _random_window(image, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed_target_window = False, random_state=None):
    random_state = check_random_state(random_state)

    # Draw a random window
    width, height = image.size

    if fixed_target_window: #if true, we don't select randomly the size of the randow window but we use target sizes instead
        crop_width = target_width
        crop_height = target_height
        #if crop_width > width or crop_height > height:
        #    print "Warning: crop larger than image"

    #Rectangular subwindows
    elif width < height:
        ratio = 1. * target_height / target_width
        min_width = min_size * width
        max_width = max_size * width

        if min_width * ratio > height:
            raise ValueError

        if max_width * ratio > height:
            max_width = height / ratio

        crop_width = min_width + random_state.rand() * (max_width - min_width)
        crop_height = ratio * crop_width

    #Square subwindows
    else:
        ratio = 1. * target_width / target_height
        min_height = min_size * height
        max_height = max_size * height

        if min_height * ratio > width:
            raise ValueError

        if max_height * ratio > width:
            max_height = width / ratio

        crop_height = min_height + random_state.rand() * (max_height - min_height)
        crop_width = ratio * crop_height

    if crop_width == 0:
        crop_width = 1
    if crop_height == 0:
        crop_height = 1

    # Draw a random position (subwindow fully contain in the image)
    px = int(random_state.rand() * (width - crop_width))
    py = int(random_state.rand() * (height - crop_height))

    # Crop subwindow
    box = (px, py, int(px + crop_width), int(py + crop_height))

    if interpolation == INTERPOLATION_NEAREST:
        pil_interpolation = Image.NEAREST
    elif interpolation == INTERPOLATION_BILINEAR:
        pil_interpolation = Image.BILINEAR
    elif interpolation == INTERPOLATION_CUBIC:
        pil_interpolation = Image.CUBIC
    elif interpolation == INTERPOLATION_ANTIALIAS:
        pil_interpolation = Image.ANTIALIAS
    else:
        pil_interpolation = Image.BILINEAR

    if fixed_target_window:
        if crop_width > width or crop_height > height:
            #subwindow larger than image, so we simply resize original image to target sizes
            sub_window = image.resize((target_width, target_height), pil_interpolation)
        else:
            sub_window = image.crop(box)

    #Rescaling of random size subwindows to fixed-size (target) using interpolation method
    else:
        sub_window = image.crop(box).resize((target_width, target_height), pil_interpolation)

    # Rotate/transpose subwindow
    # We choose randomly a right angle rotation
    if transpose:
        if np.random.rand() > 1.0 / 6:
            sub_window.transpose((Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270)[np.random.randint(5)])

    return sub_window, box

def _get_image_data(sub_window, colorspace):
    # Convert colorpace
    raw = np.array(sub_window.getdata(), dtype=np.float32)

    #print "raw ndim: %d" %raw.ndim

    if raw.ndim == 1:
        raw = raw[:, np.newaxis]

    #print "raw ndmin after newaxis: %d" %raw.ndim

    if colorspace == COLORSPACE_RGB:
        data = _raw_to_rgb(raw)
    elif colorspace == COLORSPACE_TRGB:
        data = _raw_to_trgb(raw)
    elif colorspace == COLORSPACE_HSV:
        data = _raw_to_hsv(raw)
    elif colorspace == COLORSPACE_GRAY:
        data = _raw_to_gray(raw)

    return data

# To work on images in parallel
def _partition_images(n_jobs, n_images):
    if n_jobs == -1:
        n_jobs = min(cpu_count(), n_images)

    else:
        n_jobs = min(n_jobs, n_images)

    counts = [n_images / n_jobs] * n_jobs

    for i in xrange(n_images % n_jobs):
        counts[i] += 1

    starts = [0] * (n_jobs + 1)

    for i in xrange(1, n_jobs + 1):
        starts[i] = starts[i - 1] + counts[i - 1]

    return n_jobs, counts, starts


#Output Class is the directory from which the image comes from (used in classification)
def _get_output_from_directory(target, sub_window):
    return target, sub_window.convert('RGB')

#Output class is the class of the central pixel (used in single output segmentation, see Dumont et al., 2009)
def _get_output_from_central_pixel(target, sub_window):
    assert(sub_window.mode == "RGBA")
    width, height = sub_window.size
    pixel = sub_window.getpixel(width / 2, height / 2)
    alpha = pixel[3]
    if alpha == 0:
        target = 0
    return target, sub_window.convert('RGB')

#Output classes are the classes of all output pixels (used in Segmentation, see Dumont et al., 2009)
def _get_output_from_mask(target, sub_window):
    assert(sub_window.mode == "RGBA")
    mask = np.array(sub_window.split()[3].getdata())
    y = np.zeros(mask.shape)
    y[mask == 255] = target
    return y, sub_window.convert('RGB')

#Parallel extraction of subwindows
def _parallel_make_subwindows(X, y, dtype, n_subwindows, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed, seed, verbose, get_output):
    random_state = check_random_state(seed)

    if colorspace == COLORSPACE_GRAY:
        dim = 1
    else:
        dim = 3 # default

    _X = np.zeros((len(X) * n_subwindows, dim * target_width * target_height), dtype=dtype)
    if get_output == _get_output_from_mask:
        _y = np.zeros((len(X) * n_subwindows, target_width * target_height), dtype=np.int32) #multiple output
    else :
        _y = np.zeros((len(X) * n_subwindows), dtype=np.int32) #single output


    i = 0

    for filename, target in zip(X, y):
        if verbose > 0:
            sys.stdout.write(".")
            sys.stdout.flush()

        image = Image.open(filename)

        if image.mode == "P":
            image = image.convert("RGB")

        for w in xrange(n_subwindows):
            try:
                sub_window, box = _random_window(image, min_size, max_size, target_width, target_height, interpolation, transpose, colorspace, fixed, random_state=random_state)

                output, sub_window = get_output(target, sub_window)
                data = _get_image_data(sub_window, colorspace)
                _X[i, :] = data

            except:
                print
                print "Expected dim =", _X.shape[1]
                print "Got", data.shape
                print filename
                raise

            _y[i] = output
            i += 1

    return _X, _y


class PyxitClassifier(BaseEstimator, ClassifierMixin):
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
                       get_output = _get_output_from_directory):
        self.base_estimator = base_estimator
        self.n_subwindows = n_subwindows
        self.min_size = min_size
        self.max_size = max_size
        self.target_width = target_width
        self.target_height = target_height
        self.interpolation = interpolation
        self.transpose = transpose
        self.colorspace = colorspace
        self.fixed_size = fixed_size
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.verbose = verbose
        self.get_output = get_output

        self.maxs = None

    def extract_subwindows(self, X, y, dtype=np.float32):
        # Assign chunk of subwindows to jobs
        n_jobs, _, starts = _partition_images(self.n_jobs, len(X))

        # Parallel loop
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.extract_subwindows] Extracting random subwindows"

        all_data = Parallel(n_jobs=n_jobs)(
            delayed(_parallel_make_subwindows)(
                X[starts[i]:starts[i + 1]],
                y[starts[i]:starts[i + 1]],
                dtype,
                self.n_subwindows,
                self.min_size,
                self.max_size,
                self.target_width,
                self.target_height,
                self.interpolation,
                self.transpose,
                self.colorspace,
                self.fixed_size,
                self.random_state.randint(MAX_INT),
                self.verbose,
                self.get_output)
            for i in xrange(n_jobs))

        if self.verbose > 0:
            print

        # Reduce
        _X = np.vstack(X for X, _ in all_data)
        _y = np.concatenate([y for _, y in all_data])

        return _X, _y

    def extend_mask(self, mask):
        mask_t = np.zeros(len(mask) * self.n_subwindows, dtype=np.int)

        for i in xrange(len(mask)):
            offset = mask[i] * self.n_subwindows

            for j in xrange(self.n_subwindows):
                mask_t[i * self.n_subwindows + j] = offset + j

        return mask_t

    #Build Pyxitclassifier by extracting subwindows then build of Extra-Trees (base_estimator)
    def fit(self, X, y, _X=None, _y=None):
        # Collect some data
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        y = np.searchsorted(self.classes_, y)

        # Extract subwindows
        if _X is None or _y is None:
            _X, _y = self.extract_subwindows(X, y)

        # Fit base estimator
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.fit] Building base estimator"

        print _X.shape

        self.base_estimator.fit(_X, _y)

        return self

    def predict(self, X, _X=None):
        return self.classes_.take(
            np.argmax(self.predict_proba(X, _X), axis=1),  axis=0)

    
    def predict_proba(self, X, _X=None):
        # Extract subwindows
        if _X is None:
            y = np.zeros(X.shape[0])
            _X, _y = self.extract_subwindows(X, y)

        # Predict proba
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.predict_proba] Computing class probabilities"

        y = np.zeros((X.shape[0], self.n_classes_))
        inc = 1.0 / self.n_subwindows

        try:
            _y = self.base_estimator.predict_proba(_X)

            for i in xrange(X.shape[0]):
                y[i] = np.sum(_y[i * self.n_subwindows:(i + 1) * self.n_subwindows], axis=0) / self.n_subwindows

        except:
            _y = self.base_estimator.predict(_X)

            for i in xrange(X.shape[0]):
                for j in xrange(i * self.n_subwindows, (i + 1) * self.n_subwindows):
                    y[i, _y[j]] += inc

        return y


    #Propagates subwindows into the ERT model and compute subwindow frequencies in terminal nodes 
    #ET-FL method see Maree et al., TR 2014
    def transform(self, X, _X=None, reset=False):
        # Predict proba
        if self.verbose > 0:
            print "[estimator.PyxitClassifier.transform] Transforming into leaf features"

        # Extract subwindows
        if _X is None:
            y = np.zeros(X.shape[0])
            _X, _y = self.extract_subwindows(X, y)

        # Leaf transform
        row, col, data, node_count = leaf_transform(self.base_estimator.estimators_, _X, X.shape[0], self.n_subwindows)
        __X = csr_matrix((data, (row, col)), shape=(X.shape[0], node_count), dtype=np.float32)

        # Scale features from [0, max] to [0, 1]
        if reset:
            self.maxs = None

        __X, self.maxs = inplace_csr_column_scale_max(__X, self.maxs)

        return __X
