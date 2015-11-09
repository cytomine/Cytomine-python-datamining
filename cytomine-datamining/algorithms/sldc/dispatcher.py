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


class ThyroidDispatcherAlgo(object):
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


class AreaDispatcher(ThyroidDispatcherAlgo):
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
        Tne first dispatcher algorithm (filter in cells, aggregate and
        architectural pattern)
    thyroid_dispatcher_algo : :class:`ThyroidDispatcherAlgo`
        Tne second dispatcher algorithm (identify cells in segmented aggregates)
    """

    def __init__(self, thyroid_dispatcher_algo,
                 thyroid_dispatcher_algo2):
        Dispatcher.__init__(self)
        self._algo = thyroid_dispatcher_algo
        self._sl_workflow = None
        self._algo2 = thyroid_dispatcher_algo2

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

        Parameters
        ----------
        datastore : :class:`ThyroidDataStore`
            The datastore from which to load the result of the segmentation
            and where to save the result of dispatching
        """
        dict_2_segment = {}
        # Phase 1 : simple dispatch
        for img_index, polygons in datastore.get_polygons().iteritems():
            polygons_2_sec = []
            for polygon in polygons:
                label = self._algo.dispatch(polygon)
                if label == DispatchEnum.CELL:
                    datastore.store_cell(img_index, polygon)
                elif label == DispatchEnum.AGGREGATE:
                    polygons_2_sec.append(polygon)
                elif label == DispatchEnum.ARCHITECTURAL_PATTERN:
                    datastore.store_architectural_pattern(img_index, polygon)
                    polygons_2_sec.append(polygon)
            if len(polygons_2_sec) > 0:
                dict_2_segment[img_index] = polygons_2_sec


        #Phase 2 : Redispatching
        datastore.second_segmentation()
        # Get crop from polygons (index correspond to the image)
        datastore.store_crop_to_segment(dict_2_segment)
        # Apply segmentation to crop and get new polygons of the cells
        dict_polygons = {}
#        dict_polygons = self._sl_workflow.segment_locate(datastore)
        # Filter out non-cells
        for img_index, polygons in dict_polygons.iteritems():

            for polygon in polygons:
                label = self._algo2.dispatch(polygon)
                if label == DispatchEnum.CELL:
                    datastore.store_cell(img_index, polygon)
