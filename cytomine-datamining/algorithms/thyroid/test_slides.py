# -*- coding: utf-8 -*-
import os

import cv2
from cytomine import Cytomine
from image_adapter import CytomineSlide, CytomineTileBuilder

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from slide_processing import SlideSegmenter, get_standard_kernel
from helpers.datamining.colordeconvoluter import ColorDeconvoluter

ROOT="/home/mass/GRD/r.mormont"
params = {
    'cyto_verbose' : True,
    'cyto_host' : "beta.cytomine.be",
    'cyto_ims_host' : "upload.cytomine.be",
    'cyto_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cyto_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cyto_base_path' : '/api/',
    'cyto_working_path' : '{}/nobackup/test'.format(ROOT),
}

if __name__ == "__main__":
    cytomine = Cytomine(params["cyto_host"], params["cyto_public_key"],
                        params["cyto_private_key"], base_path=params["cyto_base_path"],
                        working_path=params["cyto_working_path"], verbose=params["cyto_verbose"])

    builder = CytomineTileBuilder(cytomine)
    slide = CytomineSlide(cytomine, 186858563)
    color_deconvoluter = ColorDeconvoluter()
    color_deconvoluter.set_kernel(get_standard_kernel())
    segmenter = SlideSegmenter(color_deconvoluter)
    print "HELP"
    for i, tile in enumerate(slide.tile_iterator(builder, overlap=15, max_width=2048, max_height=2048)):
        image = tile.np_image.astype("uint8")
        segmented = segmenter.segment(image)

        impath = os.path.join(params["cyto_working_path"], "out", "image_{}.png".format(i))
        segpath = os.path.join(params["cyto_working_path"], "out", "seg_{}.png".format(i))


        print "Image in {}".format(impath)
        print "Image in {}".format(segpath)

        cv2.imwrite(impath, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imwrite(segpath, segmented)
