# -*- coding: utf-8 -*-
import os

import cv2
from PIL import Image
import numpy as np
from PIL.ImageDraw import Draw
from shapely.affinity import translate, affine_transform
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import cascaded_union
from shapely.wkt import loads
from cytomine.models import Annotation

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from adapters import CytomineAdapter, AnnotationCollectionAdapter
from helpers.datamining.rasterizer import alpha_rasterize
from helpers.datamining.colordeconvoluter import ColorDeconvoluter
from segmenters import SlideSegmenter, get_standard_kernel, AggregateSegmenter, get_standard_struct_elem
from sldc.locator import Locator
from ontology import ThyroidOntology

class Obj(object):
    pass


def mk_dataset(params, cytomine):

    # fetch annotation and filter them
    annotations = cytomine.get_annotations(id_project=params.cytomine_id_project, showMeta=True,
                                           id_user=params.cytomine_selected_users, showWKT=True)
    all_annotations = AnnotationCollectionAdapter(annotations)

    # add reviewed if requested
    if len(params.cytomine_reviewed_users) > 0:
        if len(params.cytomine_reviewed_images) > 0:
            all_annotations += cytomine.get_annotations(id_project=params.cytomine_id_project, showMeta=True, id_user=params.cytomine_reviewed_users, reviewed_only=True, id_image=params.cytomine_reviewed_images)
        else:
            all_annotations += cytomine.get_annotations(id_project=params.cytomine_id_project, showMeta=True,
                                                        id_user=params.cytomine_reviewed_users, reviewed_only=True)

    print "{} fetched annoations (pre-filter)...".format(len(all_annotations))
    # Filter annotations frm user criterion
    excluded_set = set(params.cytomine_excluded_annotations)
    excluded_terms = set(params.cytomine_excluded_terms)
    excluded_images = set(params.cytomine_excluded_images)
    filtered = [a for a in all_annotations
                if len(a.term) > 0
                    and a.id not in excluded_set
                    and set(a.term).isdisjoint(excluded_terms)
                    and a.image not in excluded_images]

    # dump annotations
    filtered = AnnotationCollectionAdapter(filtered)
    filtered = cytomine.dump_annotations(annotations=filtered, dest_path=params.pyxit_dir_ls,
                                         get_image_url_func=Annotation.get_annotation_alpha_crop_url,
                                         desired_zoom=params.cytomine_zoom_level)

    print "{} annotations dumped...".format(len(filtered))
    # make file names
    for annot in filtered:
        if not hasattr(annot, 'filename'):
            annot.filename = os.path.join(params.pyxit_dir_ls, annot.term[0], "{}_{}.png".format(annot.image, annot.id))

    return zip(*[(annot.filename, annot.term[0], annot.image, loads(annot.location)) for annot in filtered])


def circularity(polygon):
    return 4 * np.pi * polygon.area / (polygon.length * polygon.length)

if __name__ == "__main__":
    params = Obj()
    params.cytomine_host = "beta.cytomine.be"
    params.cytomine_public_key = "ad014190-2fba-45de-a09f-8665f803ee0b"
    params.cytomine_private_key = "767512dd-e66f-4d3c-bb46-306fa413a5eb"
    params.cytomine_base_path = "/api/"
    params.cytomine_zoom_level = 0
    params.cytomine_working_path = "/home/mass/GRD/r.mormont/nobackup/cv"
    params.write_path = "/home/mass/GRD/r.mormont/nobackup/area"
    params.cytomine_verbose = False
    params.cytomine_id_project = 716498
    params.cytomine_selected_users = [671279, 14]
    params.cytomine_excluded_annotations = [30675573, 18107252, 9321884, 7994253, 9313842]
    params.cytomine_excluded_terms = [9444456, 22042230, 28792193, 30559888, 15054705, 15054765] + [15109451, 15109483, 15109489, 15109495, 8844845, 8844862]
    params.cytomine_excluded_images = []
    params.cytomine_reviewed_users = []
    params.pyxit_dir_ls = "/home/mass/GRD/r.mormont/nobackup/cv/ls"

    cytomine = CytomineAdapter(params.cytomine_host, params.cytomine_public_key, params.cytomine_private_key,
                               base_path=params.cytomine_base_path, working_path=params.cytomine_working_path,
                               verbose=params.cytomine_verbose)

    # make dataset
    X, y, labels, polygons = mk_dataset(params, cytomine)

    #
    colordeconv = ColorDeconvoluter()
    colordeconv.set_kernel(get_standard_kernel())
    segmenter = SlideSegmenter(colordeconv)
    segmenter2 = AggregateSegmenter(colordeconv, get_standard_struct_elem())
    locator = Locator()

    image_map = dict()
    data = []
    i = 0
    for path, term, id_image, polygon in zip(X, y, labels, polygons):
        print polygon
        image = np.asarray(Image.open(path))
        w, h, c = image.shape
        minx, miny, _, maxy = polygon.bounds
        original_polygon = translate(polygon, -minx, -miny)
        original_polygon = affine_transform(original_polygon, [1, 0, 0, -1, 0, maxy - miny])
        print original_polygon

        # add mask to exclude external components
        if term == ThyroidOntology.CELL_INCL or term == ThyroidOntology.CELL_GLASS or \
                term == ThyroidOntology.CELL_GROOVES or term == ThyroidOntology.CELL_NORM or \
                term == ThyroidOntology.CELL_NOS or term == ThyroidOntology.CELL_POLY or \
                term == ThyroidOntology.CELL_PSEUDO:
            segmented = segmenter2.segment(image)
            polygons = locator.locate(segmented)
            polygon = cascaded_union(polygons).convex_hull

            if polygon.length == 0:
                polygon = original_polygon

        elif term == ThyroidOntology.PATTERN_PROLIF or term == ThyroidOntology.PATTERN_MINOR or \
                term == ThyroidOntology.PATTERN_NORM:
            segmented = segmenter.segment(image[:, :, 0:3])
            polygons = locator.locate(segmented)
            polygon = cascaded_union(polygons).convex_hull.intersection(original_polygon)
        if polygon.length == 0:
                polygon = original_polygon

        else:
            polygon = original_polygon

        if id_image not in image_map:
            image_map[id_image] = cytomine.get_image_instance(id_image)

        inst = image_map[id_image]
        area = polygon.area
        real = polygon.area * inst.resolution * inst.resolution
        circ = circularity(polygon)
        data.append([term, area, real, circ, 0])

        i += 1

    with open("db2.csv", "w+") as file:
        file.write("ID;area;real_area;circ;comp\n")
        for stats in data:
            file.write(";".join(["{}".format(stat) for stat in stats]))
            file.write("\n")
