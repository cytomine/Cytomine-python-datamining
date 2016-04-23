# -*- coding: utf-8 -*-
import os

import cStringIO
import timeit

import cv2
import numpy as np
import time
from PIL import Image
from cytomine import Cytomine
from image_adapter import CytomineSlide, CytomineTileBuilder

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from slide_processing import SlideSegmenter, get_standard_kernel
from helpers.datamining.colordeconvoluter import ColorDeconvoluter


ROOT="/home/mass/GRD/r.mormont"
params = {
    'cyto_verbose' : False,
    'cyto_host' : "beta.cytomine.be",
    'cyto_ims_host' : "upload.cytomine.be",
    'cyto_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cyto_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cyto_base_path' : '/api/',
    'cyto_working_path' : '{}/nobackup/test'.format(ROOT),
}




def _get_crop(cytomine, image_inst, geometry):
    """
    Download the crop corresponding to bounds on the given image instance
    from cytomine

    Parameters
    ----------
    cytomine : :class:`Cytomine`
        The cilent holding the communication
    image_inst : :class:`ImageInstance` or image instance id (int)
        The image on which to extract crop
    geometry: tuple (int, int, int, int)
        The information about the geometry of the crop structured as (offset_x, offset_y, width, height)
    """
    bounds = dict()
    bounds["x"], bounds["y"], bounds["w"], bounds["h"] = geometry
    url = image_inst.get_crop_url(bounds)
    # TODO change in the client
    url = cytomine._Cytomine__protocol + cytomine._Cytomine__host + cytomine._Cytomine__base_path + url
    resp, content = cytomine.fetch_url(url)
    if resp.status != 200:
        raise IOError("Couldn't fetch the crop for image {} and bounds {} from server (status : {}).".format(image_inst.id, geometry, resp.status))
    tmp = cStringIO.StringIO(content)
    return Image.open(tmp)


if __name__ == "__main__":
    cytomine = Cytomine(params["cyto_host"], params["cyto_public_key"],
                        params["cyto_private_key"], base_path=params["cyto_base_path"],
                        working_path=params["cyto_working_path"], verbose=params["cyto_verbose"])

    # builder = CytomineTileBuilder(cytomine)
    slide = CytomineSlide(cytomine, 186858563)
    sizes = [64, 128, 256, 400, 512, 800, 1024, 1360, 1720, 1800, 2048, 2200, 2450, 2650, 2900]
    n = 10
    times = dict()
    for size in sizes:
        for i in range(0, n):
            start = timeit.default_timer()
            image = np.asarray(_get_crop(cytomine, slide.image_instance, (0, 0, size, size)))
            stop = timeit.default_timer()
            time.sleep(0.1)
            times[size] = times.get(size, []) + [stop - start]
        print "Size {} : time {} (std: {})".format(size, np.mean(times[size]), np.std(times[size]))
