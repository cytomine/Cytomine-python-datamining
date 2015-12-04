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

from datastore import ThyroidDataStore

from cytomine.models.annotation import AlgoAnnotationTerm
from helpers.utilities.source import SlideBuffer, Image2FileSystemBuffer
from helpers.utilities.cytomineadapter import CytomineTileStreamBuilder
from helpers.utilities.cytomineadapter import CytomineCropStreamBuilder
from helpers.utilities.cytomineadapter import CropLoader
from helpers.utilities.datatype import affine_transform
from helpers.utilities import WholeSlide
from helpers.datamining.rasterizer import Rasterizer

class ThyroidCytomineDataStore(ThyroidDataStore):
    """
    ========================
    ThyroidCytomineDataStore
    ========================
    A :class:`DataStore` for the Thyorid cell classification application
    with data coming from the Cytomine platform

    Coordinate policy
    -----------------
    The polygons are stored and managed in image coordinate (origin of the
    slide, not the tile and at zoom 0) in (x, y) format where x is the column
    and y is the row (increasing downwards)
    Only upon uploading the annotations are the polygons transformed
    in cartesian coordinates.

    Constructor parameters
    ----------------------
    cytomine : :class:`Cytomine`
        The cytomine instance through which to communicate
    slide_ids : sequence of positive int
        The identifiers of the slides to process
    zoom_sl : int >= 0 (default : 1)
        The zoom level for the segment-locate part
    """

    def __init__(self, cytomine, slide_ids, zoom_sl, working_path):

        self._cytomine = cytomine

        self._slide_ids = slide_ids
        self._zoom_sl = zoom_sl
        self._slide_proxy = None
        self._img_inst = {}
        self._patterns_to_classify = []
        self._cells_to_classify = []
        self._first_segmentation = True
        self._crops_to_segment = []
        self.ss_polygons = []
        self.dict_polygons = None
        self._job = None
        self._working_path = working_path


    def _get_slides_proxy(self):
        """
        Return
        ------
        the :class:`WholeSlide` proxies corresponding to the the slide ids

        Incurs a dataflow the first time
        """
        if self._slide_proxy is None:

            self._slide_proxy = []
            for sld_id in self._slide_ids:
                # dataflow
                img_inst = self._cytomine.get_image_instance(sld_id, True)
                self._img_inst[sld_id] = img_inst
                slide = WholeSlide(img_inst)
                self._slide_proxy.append(slide)
        return self._slide_proxy


    #------------------------DataStore----------------------------#

    def second_segmentation(self):
        self._first_segmentation = False

    def get_main_slide_stream(self):
        if self._first_segmentation:
            factory = CytomineTileStreamBuilder(self._cytomine,
                                                zoom=self._zoom_sl)

            slides = self._get_slides_proxy()
            return SlideBuffer(slides, factory)
        else:
            factory = CytomineCropStreamBuilder(self._cytomine, Rasterizer())
            return SlideBuffer(self._crops_to_segment, factory)

    def store_polygons(self, dict_polygons):
        """
        Stores the given polygons in the datastore. The stored polygons will
        be scaled to zoom level 0 but will still be in image coordinates.

        Parameters
        ----------
        dict_polygons : a dictionary of polygons
                    (:class:`shapely.Polygon`)
            key = image_id => sequence of polygons
        """
        self.dict_polygons = {}
        scale_factor = pow(2, self._zoom_sl)
        scaler = affine_transform(xx_coef=scale_factor, yy_coef=scale_factor)
        for slide_id, polygons in dict_polygons.iteritems():
            tmp = []
            for polygon in polygons:
                new_polygon = scaler(polygon)
                tmp.append(new_polygon)
            self.dict_polygons[slide_id] = tmp

    #---------------ThyroidDataStore---------------------#

    def store_cell(self, img_index, polygon):
        img_inst = self._img_inst[img_index]
        self._cells_to_classify.append((img_inst, polygon))

    def store_architectural_pattern(self, img_index, polygon):
        img_inst = self._img_inst[img_index]
        self._patterns_to_classify.append((img_inst, polygon))

    def store_aggregate(self, img_index, polygon):
        raise NotImplementedError("Not yet")

    def store_crop_to_segment(self, dict_2_segment):
        """
        Parameters
        ----------
        dict_2_segment : `dict` : slide_id => sequence of polygons

        Note
        ----
        The polygons must be in the Cytomine reference system (origin in lower
        left and zoomed to correspond to the zoom0 layout)
        See :class:`ThyroidDatasotre` for full documentation.
        """
        # from slide_id => list of polygons we will go to
        # slide_id => list of pairs (polygons, bounds)
        for slide_id, polygons in dict_2_segment.iteritems():
            img_inst = self._img_inst[slide_id]
            self._crops_to_segment.append((img_inst, polygons))

    def get_cells_to_classify(self):
        factory = CropLoader(self._cytomine)
        return Image2FileSystemBuffer(self._cells_to_classify, factory, self._working_path)

    def get_arch_pattern_to_classify(self):
        factory = CropLoader(self._cytomine)
        return Image2FileSystemBuffer(self._patterns_to_classify, factory, self._working_path)

    def _upload_annotation(self, img_inst, geometry, label=None):
        image_id = img_inst.id
        # Transform in cartesian coordinates
        transfo = affine_transform(xx_coef=1,
                                   xy_coef=0,
                                   yx_coef=0,
                                   yy_coef=-1,
                                   delta_y=img_inst.height)
        geometry = transfo(geometry)

        annotation = self._cytomine.add_annotation(geometry.wkt, image_id)
        if label is not None:
            self._cytomine.add_annotation_term(annotation.id, label, label, 1.0, annotation_term_model=AlgoAnnotationTerm)

    def publish_results(self, cell_classif, arch_pattern_classif):
        # TODO make smthg more efficient with add_annotations_with_terms
        for (img_inst, polygon), label in zip(self._cells_to_classify, cell_classif):
            self._upload_annotation(img_inst, polygon, int(label))
        for (img_inst, polygon), label in zip(self._patterns_to_classify, arch_pattern_classif):
            self._upload_annotation(img_inst, polygon, int(label))

    def publish_raw_results(self):
        for slide_id in self.dict_polygons:
            for polygon in self.dict_polygons[slide_id]:
                self._upload_annotation(self._img_inst[slide_id], polygon)