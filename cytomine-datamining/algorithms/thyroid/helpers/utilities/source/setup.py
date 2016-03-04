# -*- coding: utf-8 -*-


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('source', parent_package, top_path)

    return config
