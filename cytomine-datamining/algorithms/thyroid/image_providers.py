# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

from sldc import ImageProvider, WorkflowLinker
from image_adapter import CytomineSlide


class SlideProvider(ImageProvider):
    """An image provider which generates CytomineSlides base on image instance ids
    """
    def __init__(self, cytomine, images):
        ImageProvider.__init__(self, silent_fail=True)
        self._cytomine = cytomine
        self._images = images

    def get_images(self):
        return [CytomineSlide(self._cytomine, id_img_instance) for id_img_instance in self._images]


class AggregateLinker(WorkflowLinker):
    """WorkflowLinker for extracting window images of aggregates
    """
    def __init__(self, cytomine):
        WorkflowLinker.__init__(self)
        self._cytomine = cytomine

    def get_images(self, image, workflow_info_collection):
        images = []
        for polygon, dispatch, cls in workflow_info_collection.polygons():
            if dispatch == 2:  # is aggregate
                images.append(image.window_from_polygon(polygon))
        return images
