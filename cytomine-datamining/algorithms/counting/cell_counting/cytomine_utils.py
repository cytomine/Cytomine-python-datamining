# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2017. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
import os

import cv2
import numpy as np
from cytomine.models import Annotation, AlgoAnnotationTerm

from shapely.affinity import affine_transform
from shapely.geometry import Point
from shapely.wkt import loads

from cell_counting.utils import make_dirs

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"


CROPS_PATH = "crops/"
ROIS_PATH = "rois/"
GROUNDTRUTHS_PATH = "groundtruths/"


def get_dataset(cytomine, working_path, id_project, id_term, id_roi_term, id_user=None, reviewed_only=False,
                id_roi_user=None, reviewed_only_roi=False, force_download=False):

    # Download ROI annotations
    crops_annotations = cytomine.get_annotations(id_project=id_project,
                                                 id_term=id_roi_term,
                                                 id_user=id_roi_user,
                                                 reviewed_only=reviewed_only_roi,
                                                 showWKT=True,
                                                 showMeta=True)
    # Download ROI crops
    crops = cytomine.dump_annotations(annotations=crops_annotations,
                                      dest_path=os.path.join(working_path, CROPS_PATH, str(id_project)),
                                      override=force_download,
                                      desired_zoom=0,
                                      get_image_url_func=Annotation.get_annotation_alpha_crop_url).data()

    dataset = list()
    for crop in crops:
        gt_annots = cytomine.get_annotations(id_project=id_project,
                                             id_image=crop.image,
                                             id_bbox=crop.id,
                                             id_term=id_term,
                                             id_user=id_user,
                                             reviewed_only=reviewed_only,
                                             showWKT=True).data()

        img_inst = cytomine.get_image_instance(crop.image)
        crop_location = affine_transform(loads(crop.location), [0, -1, 1, 0, img_inst.height, 0])
        offset, width, height = polygon_box(crop_location)
        affine_matrix = [0, -1, 1, 0, img_inst.height - offset[0], -offset[1]]
        gt_locations = [affine_transform(loads(gt.location), affine_matrix) for gt in gt_annots]

        groundtruth = mk_groundtruth_image(gt_locations, width, height)

        image_filename = crop.filename
        groundtruth_filename = save_groundtruth_image(groundtruth,
                                                      os.path.join(working_path, GROUNDTRUTHS_PATH, str(crop.project)),
                                                      "{}_{}.png".format(crop.image, crop.id))
        dataset.append((image_filename, groundtruth_filename))

    return zip(*dataset)


def save_groundtruth_image(groundtruth, path, filename):
    make_dirs(path)
    filename = os.path.join(path, filename)
    cv2.imwrite(filename, groundtruth * 255)
    return filename


def mk_groundtruth_image(gt_locations, width, height):
    points = [p if isinstance(p, Point) else p.centroid for p in gt_locations]
    points = [p for p in points if 0 <= p.x < height and 0 <= p.y < width]
    points = np.array([np.array([int(round(p.x)), int(round(p.y))]) for p in points]).T
    groundtruth = np.zeros((height, width))
    groundtruth[tuple(points)] = 1
    return groundtruth


def polygon_box(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    offset = (minx, miny)
    width = maxy - miny
    height = maxx - minx
    return offset, int(width), int(height)


def upload_annotations(cytomine, img, y, term=None, proba=1.):
    points = np.transpose(y.nonzero())
    points = [Point(x, y) for x, y in points]

    if isinstance(img, Annotation):
        img_inst = cytomine.get_image_instance(img.image)
        crop_location = affine_transform(loads(img.location), [0, -1, 1, 0, img_inst.height, 0])
        offset, width, height = polygon_box(crop_location)
    else:
        offset = (0, 0)
        img_inst = img

    affine_matrix = [0, 1, -1, 0, offset[1], img_inst.height - offset[0]]
    points = [affine_transform(pt, affine_matrix) for pt in points]

    for point in points:
        annotation = cytomine.add_annotation(point.wkt, img_inst.id)
        if term is not None and annotation is not None:
            cytomine.add_annotation_term(annotation.id, term, term, proba, annotation_term_model=AlgoAnnotationTerm)
