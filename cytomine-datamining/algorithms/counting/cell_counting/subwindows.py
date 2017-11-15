# -*- coding: utf-8 -*-
from functools import partial

import numpy as np

from cell_counting.image_features import FeaturesExtractor
from cell_counting.preprocessing import scoremap_transform_edt, scoremap_transform_density
from utils import open_image_with_mask, open_scoremap

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def mk_subwindows(X, y, labels, sw_input_size=(5,5), sw_output_size=(1,1), sw_extr_mode='random',
                  sw_colorspace=['RGB__rgb'], mean_radius=2, pre_transformer=None, dtype=np.float16,
                  flatten=True, dataset_augmentation=False, return_labels=False, random_state=42,
                  pre_alpha=None, sw_extr_stride=None, sw_extr_ratio=None,
                  sw_extr_score_thres=None, sw_extr_npi=None):
    """
    Extract subwindows from provided data.

    Parameters
    ----------
    X: array-like of str
        The list of input image filenames.
    y: array-like of str
        The list of binary mask filenames.
    labels: array-like of int, None
        The list of input labels.
    sw_input_size: tuple of two int
        The size (width, height) of input subwindows.
    sw_output_size: tuple of two int
        The size (width, height) of output subwindows.
    sw_extr_mode: {'random', 'scoremap_constrained', 'sliding'}
        The mode of extraction for input suwbindows.
    sw_colorspace: list of str
        The list of colorspaces from which subwindows are extracted.
    mean_radius: int
        The mean radius of objects
    pre_transformer: {'euclidean_distance_transform, 'edt', 
                           'proximity', density'}, optional (default=None)
        The type of scoremap transformer required.
    dtype: data-type, optional (default=np.float16)
        The data-type of extracted subwindow values.
    dataset_augmentation: bool, optional (default=False)
        If dataset augmentation must be performed.
    return_labels: bool, optional (default=False)
        If corresponding label must be returned for each extracted subwindow.
    random_state: int, optional (default=42)
        An optional seed to make random number generator predictable.

    Returns
    -------
    _X: array-like of shape (n_subwindows, input_width * input_height * n_features)
        The input subwindows.
    _y: array-like of shape (n_subwindows, output_width * output_height)
        The output subwindows.
    
    Notes
    -----
    _labels: array of shape (n_subwindows,), optional: only if `return_labels`=True
        The labels corresponding to subwindows.
    """
    labels = [0] * len(y) if labels is None else labels

    _X, _y, _labels = list(), list(), list()
    for image_filename, groundtruth_filename, label in zip(X, y, labels):
        image, mask = open_image_with_mask(image_filename, padding=half_size(sw_input_size))
        scoremap = open_scoremap(groundtruth_filename, padding=half_size(sw_input_size))

        if pre_transformer in ('euclidean_distance_transform', 'edt'):
            if pre_alpha is None:
                raise ValueError('alpha parameter required/invalid')
            scoremap = scoremap_transform_edt(scoremap, mean_radius, pre_alpha)
        elif pre_transformer == 'density':
            scoremap = scoremap_transform_density(scoremap, mean_radius)

        image_filters = FeaturesExtractor(sw_colorspace, dtype=dtype).build(image).feature_image
        __X, __y = extract_subwindows_image(image_filters, scoremap, mask,
                                            sw_input_size, sw_output_size,
                                            sw_extr_mode, mean_radius, flatten,
                                            dataset_augmentation, random_state,
                                            sw_extr_stride, sw_extr_ratio,
                                            sw_extr_score_thres, sw_extr_npi)
        _X.append(__X)
        _y.append(__y)

        if return_labels:
            _labels.append([label] * len(__y))

    _X = np.vstack(([x for x in _X])).squeeze()
    _y = np.vstack(([y for y in _y])).squeeze()
    _labels = np.array(_labels, dtype=np.int16).squeeze()

    if return_labels:
        return _X, _y, _labels
    else:
        return _X, _y


def extract_subwindows_image(image, scoremap, mask, input_window_size, output_window_size, mode, mean_radius,
                             flatten=True, dataset_augmentation=False, random_state=42, sw_extr_stride=None,
                             sw_extr_ratio=None, sw_extr_score_thres=None, sw_extr_npi=None):
    """
    Extract subwindows from the multi-spectral provided image.

    Parameters
    ----------
    image: array-like of shape (width, height, n_features)
        The multi-spectral image.
    scoremap: array-like of shape (width, height)
        The corresponding scoremap.
    mask: array-like of shape (width, height)
        The corresponding mask.
    input_window_size: tuple of two int
        The size (width, height) of input subwindows.
    output_window_size: tuple of two int
        The size (width, height) of output subwindows.
    mode: {'random', 'scoremap_constrained', 'sliding'}
        The mode of extraction for input suwbindows.
    mean_radius: int
        The mean radius of objects
    dataset_augmentation: bool, optional (default=False)
        If dataset augmentation must be performed.
    random_state: int, optional (default=42)
        An optional seed to make random number generator predictable.

    Returns
    -------
    X: array-like of shape (n_subwindows, input_width * input_height * n_features)
        The input subwindows.
    y: array-like of shape (n_subwindows, output_width * output_height)
        The output subwindows.
    """
    input_window_size_half = half_size(input_window_size)
    output_window_size_half = half_size(output_window_size)

    if dataset_augmentation:
        np.random.seed(random_state)
        methods = [np.fliplr, np.flipud, np.rot90,
                   partial(np.rot90, k=2), partial(np.rot90, k=3)]
    else:
        methods = []

    if mode == 'random':
        if sw_extr_npi is None:
            raise ValueError('number_per_image parameter required/invalid')
        window_centers = _extract_random(mask, sw_extr_npi)
    elif mode == 'scoremap_constrained':
        if sw_extr_ratio is None:
            raise ValueError('bg_ratio parameter required/invalid')
        if sw_extr_score_thres is None:
            raise ValueError('score_threshold required/invalid')
        window_centers = _extract_scoremap_constrained(mask, scoremap, sw_extr_ratio,
                                                       sw_extr_score_thres)
    else:
        raise ValueError('unknown mode')

    X, y = list(), list()
    for window_center in window_centers:
        top, right, bottom, left = subwindow_box(input_window_size,
                                                 input_window_size_half,
                                                 window_center)
        input_window = image[slice(top, bottom), slice(left, right), :]

        top, right, bottom, left = subwindow_box(output_window_size,
                                                 output_window_size_half,
                                                 window_center)
        output_window = scoremap[slice(top, bottom), slice(left, right)]

        if flatten:
            X.append(input_window.ravel())
            y.append(output_window.ravel())
        else:
            X.append(input_window)
            y.append(output_window)

        # TODO
        if dataset_augmentation:
            for method in methods:
                X.append(method(input_window).ravel())
                if output_window.ndim > 1:
                    y.append(method(output_window).ravel())
                else:
                    y.append(output_window.ravel())

    del window_centers
    return np.array(X), np.array(y)


def _extract_random(mask, number_per_image):
    """
    Extract `number_per_image` positions at random in `mask`.

    Parameters
    ----------
    mask: array-like of shape (width, height)
        The image mask.
    number_per_image: int 
        The number of positions to extract.

    Returns
    -------
    positions: list of 2 int tuple
        The list of positions.
    """
    mask = np.asarray(mask)
    idxs = np.argwhere(mask > 0)
    number_per_image = min(number_per_image, idxs.shape[0])
    idxs = idxs[np.random.choice(idxs.shape[0], number_per_image, replace=False)]
    return [tuple(idx) for idx in idxs]


def _extract_scoremap_constrained(mask, scoremap, bg_ratio, score_threshold):
    """
    Extract all positions in mask that correspond to a score higher than given
    score_threshold in scoremap and add a ratio of `bg_ratio` of random
    positions whose score is less than score_threshold.

    Parameters
    ----------
    mask: array-like of shape (width, height)
        The image mask.
    scoremap: array-like of shape (width, height)
        The scoremap mask.
    bg_ratio: float
        The ratio of random positions to add.
    score_threshold: float
        The score above which all positions are extracted.

    Returns
    -------
    positions: list of 2 int tuple
        The list of positions.
    """
    sc_norm = scoremap / np.max(scoremap)
    foreground_idxs = np.argwhere(np.logical_and(sc_norm >= score_threshold, mask > 0))
    background_idxs = np.argwhere(np.logical_and(sc_norm < score_threshold, mask > 0))

    # Balance dataset : Randomly select 'n_additional_sw' subwindows
    # whose center is black or grey to complete dataset.
    n_additional_sw = min(int(bg_ratio * foreground_idxs.shape[0]), background_idxs.shape[0])

    if n_additional_sw > 0:
        background_idxs = background_idxs[np.random.choice(background_idxs.shape[0], n_additional_sw, replace=False)]
    else:
        background_idxs = []

    return [tuple(idx) for idx in foreground_idxs] + [tuple(idx) for idx in background_idxs]


def subwindow_box(size, half_size, center):
    """
    Compute the bounding box coordinates of a subwindow.

    Parameters
    ----------
    size: tuple of two integers 
        The size (width, height) of the subwindow.
    half_size: tuple of two integers
        The half size (half_width, half_height) of the subwindow.
    center: tuple of two integers
        The position of the subwindow center (x, y).

    Returns
    -------
    top: integer
        The top position (along first axis)
    right: integer
        The right position (along second axis)
    bottom: integer
        The bottom position (along first axis)
    left: integer
        The left position (along second axis)
    """

    top = center[0] - half_size[0]
    bottom = top + size[0]
    left = center[1] - half_size[1]
    right = left + size[1]
    return top, right, bottom, left


def half_size(size):
    """
    Compute the half size of a subwindow.

    Parameters
    ----------
    size: tuple of two integers
        The size (width, height) of the subwindow.

    Returns
    -------
    half_size: tuple of two integers
        The half size (half_width, half_height) of the subwindow.
    """

    return tuple(s // 2 for s in size)


def all_subwindows_generator(image, mask, sw_input_size, sw_colorspace,
                             dtype=np.float16, batch_size=100):
    """
    Generate all input subwindows at every possible position in the provided mask
    by batch of size `batch_size.

    Parameters
    ----------
    image: array-like of shape (width, height, 3)
        The image.
    mask: array-like of shape (width, height)
        The image mask.
    sw_input_size: tuple of two int
        The size (width, height) of input subwindows.
    sw_colorspace: list of str
        The list of colorspaces from which subwindows are extracted.
    dtype: data-type, optional (default=np.float16)
        The data-type of extracted subwindow values.
    batch_size: int, optional (default=100)
        The batch size.

    Yields
    -------
    batch_subwindows: list of array-like 
                      of shape (batch_size, input_width * input_height * n_features)
        The input subwindows.
    batch_coords: list of 2 int tuple
        The corresponding center coordinates
    """
    input_window_size_half = half_size(sw_input_size)
    image_filters = FeaturesExtractor(sw_colorspace, dtype=dtype).build(image).feature_image
    window_centers = np.argwhere(np.asarray(mask) > 0)

    batch_subwindows = []
    batch_coords = []
    for i, window_center in enumerate(window_centers):
        top, right, bottom, left = subwindow_box(sw_input_size, input_window_size_half, window_center)
        batch_subwindows.append(image_filters[slice(top, bottom), slice(left, right), :].ravel())
        batch_coords.append(window_center)

        if i % batch_size == 0 or i == len(window_centers) - 1:
            yield batch_subwindows, batch_coords
            batch_subwindows = []
            batch_coords = []
