# -*- coding: utf-8 -*-
from shapely.affinity import translate
from sldc import ImageProvider, WorkflowExecutor, PolygonTranslatorWorkflowExecutor, SilentLogger
from image_adapter import CytomineSlide, CytomineMaskedWindow

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class SlideProvider(ImageProvider):
    """An image provider which generates CytomineSlides base on image instance ids
    """
    def __init__(self, cytomine, images, logger=SilentLogger()):
        ImageProvider.__init__(self, silent_fail=True, logger=logger)
        self._cytomine = cytomine
        self._images = images

    def get_images(self):
        return [CytomineSlide(self._cytomine, id_img_instance) for id_img_instance in self._images]


class AggregateWorkflowExecutor(PolygonTranslatorWorkflowExecutor):
    """WorkflowExecutor for extracting window images of aggregates and then translating generated polygons
    """
    def __init__(self, cytomine, workflow, logger=SilentLogger()):
        WorkflowExecutor.__init__(self, workflow, logger=logger)
        self._cytomine = cytomine

    def get_images(self, image, workflow_info_collection):
        if len(workflow_info_collection) != 1:
            raise RuntimeError("One execution expected, got {}.".format(len(workflow_info_collection)))
        workflow_information = workflow_info_collection[0]
        images = []
        for polygon, dispatch, cls in workflow_information.iterator():
            if dispatch == 1:  # is aggregate
                image_window = image.window_from_polygon(polygon)
                trans_poly = translate(polygon, -image_window.offset_x, -image_window.offset_y)
                masked_image = CytomineMaskedWindow.from_window(image_window, trans_poly)
                images.append(masked_image)
        return images
