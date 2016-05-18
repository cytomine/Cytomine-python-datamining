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

import workflow
from sldc import Logger

if __name__ == "__main__":
    args = list()
    args.append("--cell_classifier")
    args.append("/home/mass/GRD/r.mormont/models/validated/incl_short.pkl")
    args.append("--aggregate_classifier")
    args.append("/home/mass/GRD/r.mormont/models/validated/prolif_short.pkl")
    args.append("--dispatch_classifier")
    args.append("/home/mass/GRD/r.mormont/models/validated/cpo_short.pkl")
    args.append("--host")
    args.append("beta.cytomine.be")
    args.append("--public_key")
    args.append("ad014190-2fba-45de-a09f-8665f803ee0b")
    args.append("--private_key")
    args.append("767512dd-e66f-4d3c-bb46-306fa413a5eb")
    args.append("--software_id")
    args.append("152714969")
    args.append("--project_id")
    args.append("186829908")
    args.append("--slide_ids")
    args.append("186859011")  # 186859011,186858563,186851426,186851134,186850855,186850602,186850322,186849981,186849450,186848900,186848552,186847588,186847313")
    args.append("--tile_max_height")
    args.append("2048")
    args.append("--tile_max_width")
    args.append("2048")
    args.append("--working_path")
    args.append("/home/mass/GRD/r.mormont/nobackup/launch/")
    args.append("--base_path")
    args.append("/api/")
    args.append("--verbose")
    args.append("{}".format(Logger.INFO))
    args.append("--n_jobs")
    args.append("4")
    workflow.main(args)
