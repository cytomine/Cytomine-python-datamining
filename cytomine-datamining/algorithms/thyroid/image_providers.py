# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from sldc import ImageProvider, WorkflowLinker
from image_adapter import CytomineSlide


class SlideProvider(ImageProvider):

    def __init__(self, cytomine, images):
        ImageProvider.__init__(self, silent_fail=True)
        self._cytomine = cytomine
        self._images = images

    def get_images(self):
        return [CytomineSlide(self._cytomine, id_img_instance) for id_img_instance in self._images]


class AggregateLinker(WorkflowLinker):

    def __init__(self, cytomine, cluster_rule, aggregate_rule):
        WorkflowLinker.__init__(self)
        self._cytomine = cytomine
        self._cluster_rule = cluster_rule
        self._aggregate_rule = aggregate_rule

    def get_images(self, image, polygons_classes):
        images = []
        for polygon, cls in polygons_classes:
            if self._to_process(polygon):
                minx, miny, maxx, maxy = polygon.bounds
                offset = (minx, miny)
                width = maxx - minx
                height = maxy - miny
                images.append(image.window(offset, width, height))
        return images

    def _to_process(self, polygon):
        return self._cluster_rule(polygon) or self._aggregate_rule(polygon)
