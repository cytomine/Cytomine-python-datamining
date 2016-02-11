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
PROJECTID = "186829908"
SLIDEIDS = [#"186859011", "186858563", "186851426", "186851134", "186850855", "186850602", "186850322", "186849981",
            "186849450", "186848900", "186848552" ]#, "186847588"] #"186847313", "186845954", "186845730", "186845571",
           # "186845377", "186845164", "186844820", "186844344", "186843839", "186843325", "186842882", "186842285",
           # "186842002", "186841715", "186841154", "186840535", "186836213"]
MODEL_PATH = "/home/vagrant/models"
ARCH_CLASS = MODEL_PATH + "/arch_model_better.pkl"
CELL_CLASS = MODEL_PATH + "/cell_model_better.pkl"
EARLY_STOPPING = "False"

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
