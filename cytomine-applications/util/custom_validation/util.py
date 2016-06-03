# -*- coding: utf-8 -*-
import os

import shutil

from cytomine.models import Annotation

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def str2list(l, conv=int):
    return [conv(v) for v in l.split(",")]


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

