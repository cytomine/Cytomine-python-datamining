# -*- coding: utf-8 -*-
import inspect
import numpy as np
import cv2
import os
import copy

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def check_params(fns, params, exceptions=None):
    if not exceptions:
        exceptions = []

    legal_params = []
    for fn in fns:
        legal_params += inspect.getargspec(fn)[0]
    legal_params = set(legal_params)

    for params_name in params:
        if params_name not in legal_params:
            if params_name not in exceptions:
                raise ValueError(
                        '{} is not a legal parameter'.format(params_name))


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def files_in_directory(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            and not f.startswith(".")]


def make_dirs(path, remove_filename=False):
    if remove_filename:
        path = os.path.dirname(path)

    if not os.path.exists(path):
        os.makedirs(path)


def pad_image(image, padding=(0,0), mode='reflect'):
    if padding == (0,0):
        return image

    if image.ndim == 3:
        pad = ((padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
    else:
        pad = ((padding[0], padding[0]), (padding[1], padding[1]))

    return np.pad(image, pad, mode=mode)


def open_image(filename, flag=None, padding=(0,0), padding_mode='reflect'):
    if not os.path.exists(filename):
        raise IOError("File " + filename + " does not exist !")

    if flag in ('L', cv2.IMREAD_GRAYSCALE):
        flag = cv2.IMREAD_GRAYSCALE
    elif flag in ('RGB', cv2.IMREAD_COLOR):
        flag = cv2.IMREAD_COLOR
    else:
        flag = cv2.IMREAD_UNCHANGED

    im = cv2.imread(filename, flag)

    if im is None:
        raise ValueError("Image " + filename + "unreadable !")

    return pad_image(im, padding, padding_mode)


def open_image_with_mask(filename, padding=(0, 0), splitted=True):
    image = open_image(filename)
    if image.shape[2] == 3:
        mask = pad_image(np.ones_like(image[:, :, 0]), padding=padding, mode='constant')
        image = pad_image(image, padding=padding, mode='reflect')
    elif image.shape[2] == 4:
        mask = pad_image(image[:, :, 3], padding=padding, mode='constant')
        image = pad_image(image[:, :, :3], padding=padding, mode='reflect')
    else:
        raise ValueError("Impossible to load image")

    if not splitted:
        return np.dstack((image, mask))
    else:
        return image, mask


def open_scoremap(filename, padding=(0, 0)):
    scoremap = open_image(filename, flag='L', padding=padding)
    scoremap /= np.max(scoremap)
    return scoremap.astype(np.float16)


def check_default(param, default_value=None, return_list=True):
    if param is None or len(param) == 0:
        if return_list:
            return [default_value]
        else:
            return default_value

    return param


def check_max_features(max_features):
    ret = []
    for mf in max_features:
        if '.' in mf:
            ret.append(float(mf))
        elif mf == 'sqrt':
            ret.append(mf)
        else:
            ret.append(int(mf))

    return ret


def params_remove_list(params):
    param_names = [p for p in dir(params) if not p.endswith('__')]

    for param_name in param_names:
        v = getattr(params, param_name, None)
        if isinstance(v, list):
            setattr(params, param_name, v[0])

    return params


def params_remove_none(params):
    ret = copy.deepcopy(params)
    param_names = [p for p in dir(ret) if not p.endswith('__')]

    for param_name in param_names:
        v = getattr(ret, param_name, None)
        if v is None:
            setattr(ret, param_name, " ")

    return ret
