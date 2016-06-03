# -*- coding: utf-8 -*-

"""
Copyright 2010-2013 University of LiÃ¨ge, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of LiÃ¨ge, Belgium"
__version__ = '0.1'


from abc import ABCMeta, abstractmethod
import cv2
from shapely.geometry.polygon import Polygon
from ..utilities.datatype import affine_transform

class Locator:
    """
    =======
    Locator
    =======
    A :class:`Locator` is charged of vectorizing polygons from an image.
    Polygons are pixels whose value exceed zero

    Note
    ----
    Beware that the vectorization procedure may perform structural
    generalization and alter somewhat the position
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def vectorize(self, np_image, offset=None):
        """
        Vectorize the polygons in the given images

        Parameters
        ----------
        np_image : numpy.ndarray representation of images
            The binary image to process. Background must be black (value of
            zero)
        offset : tuple(row_offset, col_offset) (Default : None)
            if not None, apply the given offset to the vectorized polygons

        Return
        ------
        polygons : sequence of :class:`shapely.Polygon` with (x, y)
        coordinate system where x is the column and y is the row
        (increasing downwards)
            The polygons found in the image
        """
        pass


class CV2Locator(Locator):
    """
    ==========
    CV2Locator
    ==========
    A :class:`CV2Locator` is a locator implementing the vectorization routine
    through openCV. The procedure will alter the geometry
    """

    def vectorize(self, np_image, offset=None):
        #borrowed from cytomine_utilities.objectfinder (v 1.0)
        #CV_RETR_EXTERNAL to only get external contours.
        contours, hierarchy = cv2.findContours(np_image.copy(),
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Note: points are represented as (col, row)-tuples apparently
        transform = lambda x:x
        if offset is not None:
            row_off, col_off = offset
            transform = affine_transform(delta_x=col_off, delta_y=row_off)
        components = []
        if len(contours) > 0:
            top_index = 0
            tops_remaining = True
            while tops_remaining:
                exterior = contours[top_index][:, 0, :].tolist()

                interiors = []
                # check if there are childs and process if necessary
                if hierarchy[0][top_index][2] != -1:
                    sub_index = hierarchy[0][top_index][2]
                    subs_remaining = True
                    while subs_remaining:
                        interiors.append(contours[sub_index][:, 0, :].tolist())

                        # check if there is another sub contour
                        if hierarchy[0][sub_index][0] != -1:
                            sub_index = hierarchy[0][sub_index][0]
                        else:
                            subs_remaining = False

                # add component tuple to components only if exterior is a polygon
                if len(exterior) > 3:
                    polygon = Polygon(exterior, interiors)
                    polygon = transform(polygon)
                    components.append(polygon)

                # check if there is another top contour
                if hierarchy[0][top_index][0] != -1:
                    top_index = hierarchy[0][top_index][0]
                else:
                    tops_remaining = False

        del contours
        del hierarchy
        return components
