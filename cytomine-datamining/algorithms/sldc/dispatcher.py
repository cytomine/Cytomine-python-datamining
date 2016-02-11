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


import numpy as np

class Dispatcher(object):
    """
    ==========
    Dispatcher
    ==========
    A :class:`Dispatcher` takes some information from the workflow's
    :class:`DataStore` to store it elsewhere (possibly in the same
    :class:`DataStore`)

    """
    def __init__(self):
        pass

    def dispatch(self, datastore):
        """
        Performs the dispatching routine.

        By default, this function does nothing. It must be reimplemented
        if need be.

        Parameters
        ----------
        datastore : :class:`DataStore`
            The :class:`DataStore` from which to retrieve the information to
            dispatch
        """
        pass


class DispatchEnum:
    CELL = 1
    AGGREGATE = 2
    ARCHITECTURAL_PATTERN = 3
    ARTIFACT = 4


class PolygonDispatcher(object):
    """
    =====================
    ThyroidDispatcherAlgo
    =====================
    A prodecude to dispatch the polygons
    """

    def dispatch(self, polygon):
        """
        Dispatch the polygon in one of the four categories : cell, aggregate,
        architectural pattern or artifact

        Return
        ------
        label : int in {DispatchEnum.CELL, DispatchEnum.AGGREGATE,
                        DispatchEnum.ARCHITECTURAL_PATTERN,
                        DispatchEnum.ARTIFACT}
            The dispatching label
        """
        pass


class AreaDispatcher(PolygonDispatcher):
    """
    ==============
    AreaDispatcher
    ==============
    A :class:`ThyroidDispatcherAlgo` which dispatches base on the polygon
    area (and circularity)

    Constructor parameters
    ----------------------
    cell_min_area : float
        The cells minimum area. It must be consistent with the polygon
        coordinate system. In particular with the scale
    cell_max_area : float
        The cells maximum area. It must be consistent with the polygon
        coordinate system. In particular with the scale
    cell_min_circularity : float
        The cells minimum circularity. It must be consistent with the polygon
        coordinate system. In particular with the scale
    cluster_min_cell_nb : int
        The minimum number of cells to form a cluster. It must be consistent
        with the polygon coordinate system. In particular with the scale
    """
    def __init__(self,
                 cell_min_area,
                 cell_max_area,
                 cell_min_circularity,
                 cluster_min_cell_nb):

        self.cell_min_area = cell_min_area
        self.cell_max_area = cell_max_area
        self.cell_min_circularity = cell_min_circularity
        self.cluster_min_cell_nb = cluster_min_cell_nb

    def dispatch(self, polygon):
        p = polygon
        if p.area < self.cell_min_area:
            return DispatchEnum.ARTIFACT

        circularity = 4*np.pi*p.area / (p.length*p.length)
        if self.cell_min_area <= p.area <= self.cell_max_area:
            if circularity > self.cell_min_circularity:
                return DispatchEnum.CELL

        elif self.cell_max_area < p.area < self.cluster_min_cell_nb*self.cell_max_area:
            return DispatchEnum.AGGREGATE
        else:
            return DispatchEnum.ARCHITECTURAL_PATTERN


class ThyroidDispatcher(Dispatcher):
    """
    =================
    ThyroidDispatcher
    =================
    A :class:`Dispatcher` for the Thyroid cell classification application.

    Constructor parameters
    ----------------------
    thyroid_dispatcher_algo : :class:`ThyroidDispatcherAlgo`
        Tne dispatcher algorithm
    """

    def __init__(self, thyroid_dispatcher_algo):
        Dispatcher.__init__(self)
        self._algo = thyroid_dispatcher_algo

    def dispatch(self, datastore):
        """
        Dispatch the segmented polygons from the previous stage into
        - cells
        - aggregates
        - architectural pattern
        The dispatching is used using the dispatching algorithm provided by the user
        Parameters
        ----------
        datastore : :class:`ThyroidDataStore`
            The datastore from which to load the result of the segmentation
            and where to save the result of dispatching
        """
        # Phase 1 : simple dispatch
        for img_index, polygons in datastore.get_polygons().iteritems():
            for polygon in polygons:
                label = self._algo.dispatch(polygon)
                self._save_polygon(datastore, label, img_index, polygon)

    def _save_polygon(self, datastore, label, img_index, polygon):
        """
        Saves a polygon in a datastore according to its dispatch label.

        Parameters
        ----------
        datastore : :class:`ThyroidDataStore`
            The datastore from which to load the result of the segmentation
            and where to save the result of dispatching
        label: :enum:`DispatchEnum`
            The predicted label for the given polygon
        img_index: int
            The index of the image in which is located the polygon
        polygon:
            The polygon to store
        """
        if label == DispatchEnum.CELL:
            datastore.store_cell(img_index, polygon)
        elif label == DispatchEnum.ARCHITECTURAL_PATTERN:
            datastore.store_architectural_pattern(img_index, polygon)


class FirstPassThyroidDispatcher(ThyroidDispatcher) :
    """
    =================
    FirstPassThyroidDispatcher
    =================
    A :class:`Dispatcher` for the Thyroid cell classification application.
    In addition to the dispatching, this dispatcher saves aggregate and architectural patterns into the datastore
    for further segmentation.

    Constructor parameters
    ----------------------
    dispatcher_algo: :class:`ThyroidDispatcherAlgo`
        Tne dispatcher algorithm
    """

    def __init__(self, dispatcher_algo):
        ThyroidDispatcher.__init__(self, dispatcher_algo)
        self._sl_workflow = None

    def set_sl_workflow(self, sl_workflow):
        """
        Set the segment/locate workflow instance to use for the second
        segmentation.

        It *must* be set !

        Parameters
        ----------
        sl_workflow : :class:`SLDCWorflow`
            The instance in charge of the segmentation and localization
        """
        self._sl_workflow = sl_workflow

    def dispatch(self, datastore):
        """
        Dispatch the segmented polygons from the previous stage into
        - cells
        - aggregates
        - architectural pattern

        The aggregates and architectural patterns are saved into the datastore using store_crop_to_segment for
        further segmentation.

        Parameters
        ----------
        datastore : :class:`ThyroidDataStore`
            The datastore from which to load the result of the segmentation
            and where to save the result of dispatching
        """
        dict_2_segment = {}

        for img_index, polygons in datastore.get_polygons().iteritems():
            arch_or_aggr = []
            for polygon in polygons:
                label = self._algo.dispatch(polygon)
                self._save_polygon(datastore, label, img_index, polygon)

                # check whether the polygons must be saved for second segmentation
                if label == DispatchEnum.AGGREGATE or label == DispatchEnum.ARCHITECTURAL_PATTERN:
                    arch_or_aggr.append(polygon)

            if len(arch_or_aggr) > 0:
                dict_2_segment[img_index] = arch_or_aggr

        datastore.store_crop_to_segment(dict_2_segment)