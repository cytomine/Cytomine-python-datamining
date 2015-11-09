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

SOFTWAREID = "152714969"
PROJECTID = "151860018"
#SLIDEIDS = ["151870700"]
#SLIDEIDS = ["151870700", "151870615"]
SLIDEIDS = ["151870700", "151870615", "151870539", "151870465", "151870433",
            "151870321", "151870170", "151870070", "151869994", "151869936"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-v" "--verbose", help="increase output verbosity",
    #                 action="store_true")
    parser.add_argument("public_key",
                        help="User public key")
    parser.add_argument("private_key",
                        help="User Private key")

    keys = parser.parse_args()

    args = []
    args.append("cell_classif")
    args.append("arch_pattern_classif")
    args.append("beta.cytomine.be")
    args.append(keys.public_key)
    args.append(keys.private_key)
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


