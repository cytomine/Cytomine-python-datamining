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

import argparse

from progressmonitor import dict_config

import thyroidapp

# TODO replace accordingly
from dummyclassifier import DummyClassifier


HOST = "beta.cytomine.be"
PUBLICKEY = "ad014190-2fba-45de-a09f-8665f803ee0b"
PRIVATEKEY = "767512dd-e66f-4d3c-bb46-306fa413a5eb"
SOFTWAREID = "152714969"
PROJECTID = "151860018"
SLIDEIDS = ["151870700"]
MODEL_PATH = "/home/vagrant/models"
ARCH_CLASS = MODEL_PATH + "/arch_model.pkl"
CELL_CLASS = MODEL_PATH + "/cell_model.pkl"

if __name__ == "__main__":

    args = list()
    args.append(CELL_CLASS)
    args.append(ARCH_CLASS)
    args.append(HOST)
    args.append(PUBLICKEY)
    args.append(PRIVATEKEY)
    args.append(SOFTWAREID)
    args.append(PROJECTID)
    for slide_id in SLIDEIDS:
        args.append(slide_id)

    config = {
        "version": 1,
        "generator_monitors": {
            "CM.SLDC": {
                "format_str": "{$task} {$progressbar} {$time} {$exception}",
                "rate": 0.1,
                "callback_factory": "$stdout",
                "period": 5,
            },
        },
        "code_monitors": {
            "C_CM.SLDC": {
                "format_str": "{$task} {$elapsed} {$exception}",
                "callback_factory": "$stdout"
            },

        },
    }

    dict_config(config)

    thyroidapp.main(args)


