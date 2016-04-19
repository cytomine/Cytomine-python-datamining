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


HOST = "beta.cytomine.be"
PUBLICKEY = "ad014190-2fba-45de-a09f-8665f803ee0b"
PRIVATEKEY = "767512dd-e66f-4d3c-bb46-306fa413a5eb"
SOFTWAREID = "152714969"
PROJECTID = "186829908"
SLIDEIDS = [ "186859011", "186858563", "186851426", "186851134", "186850855", "186850602", "186850322", "186849981",
             "186849450", "186848900", "186848552", "186847588", "186847313", "186845954", "186845730", "186845571",
             "186845377", "186845164", "186844820", "186844344", "186843839", "186843325", "186842882", "186842285",
             "186842002", "186841715", "186841154", "186840535" ]
MODEL_PATH = "/home/mass/GRD/r.mormont/models"
ARCH_CLASS = MODEL_PATH + "/patterns_prolif_vs_norm.pkl"
CELL_CLASS = MODEL_PATH + "/cells_inclusion_vs_norm.pkl"
DISP_CELL = MODEL_PATH + "/cells_reduced_vs_all.pkl"
DISP_ARCH = MODEL_PATH + "/patterns_vs_all.pkl"

if __name__ == "__main__":

    args = list()
    args.append(CELL_CLASS)
    args.append(ARCH_CLASS)
    args.append(DISP_CELL)
    args.append(DISP_ARCH)
    args.append(HOST)
    args.append(PUBLICKEY)
    args.append(PRIVATEKEY)
    args.append(SOFTWAREID)
    args.append(PROJECTID)
    for slide_id in SLIDEIDS:
        args.append(slide_id)

    workflow.main(args)
