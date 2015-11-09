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


from progressmonitor import monitor_with
from ..utilities.taskmanager import SerialExecutor

class MiningWorkflow(object):
    """
    ==============
    MiningWorkflow
    ==============
    A :class:`MiningWorkflow` is a datamining processing workflow. It
    exhibits logging facilities and in particular progression logging
    (see :class`Progressable` for more information) and contains a
    :class:`TaskExecutor` so as to allow for potential parallelization
    of tasks.

    Constructor parameters
    ----------------------
    task_executor : :class:`TaskExecutor` (default : SerialExecutor())
        The :class:`TaskExecutor` to use. By default, the executor is serial
        an no parallelization occurs
    """

    def __init__(self, task_executor=SerialExecutor()):
        self.__executer = task_executor

    def get_executor(self):
        """
        Return
        ------
        task_executor : :class:`TaskExecutor
            The instance :class:`TaskExecutor
        """
        return self.__executer

    def submit_to_executor(self, function, data, *args, **kwargs):
        """
        Delegates to the instance :class:`TaskExecutor` (see the
        :class:`TaskExecutor` doc for more information)
        """
        return self.__executer.execute(function, data, *args, **kwargs)


class SLDCWorkflow(MiningWorkflow):
    """
    ===============
    SLDCCoordinator
    ===============
    A :class:`SLDCCoordinator` coordinates a SLDC datamining workflow :
    Segment : find the intersting part of a image for the task at hand
    Locate : extract coordinates of the intersting part
    Dispatch : sort interesting part in groups based on their characteristics
    Classify : perform fine-grain classification on each group

    Note
    ----
    It is the responsibility of the user to build a coherent SLDC by choosing
    appropriately components which works together. In particular, the trio
    dispatcher - datastore - classifier.

    Constructor parameters
    ----------------------
    tile_filter : :class:`Filter`
        The filtering object procedure to decide whether to treat a tile or not
    segmenter : :class:`Segmenter`
        The segmentation object procedure to segment the accepted tiles
    locator : :class:`Locator`
        The segmentation object procedure to extract polygons covering the
        segmented objects
    merger_builder : :class:`MergerFactory`
        The factory for the :class:`Merger`
    dispatcher : :class:`Dispatcher`
        The dispatcher
    classifier : :class:`Classifier`
        The final classifier
    task_executor : :class:`TaskExecutor`
        A :class:`TaskExecutor` for the segment and locate part
    """

    def __init__(self, tile_filter, segmenter, locator,
                 merger_builder, dispatcher, classifier,
                 task_executor=SerialExecutor()):

        MiningWorkflow.__init__(self, task_executor)
        self.filter = tile_filter
        self.segmenter = segmenter
        self.locator = locator
        self.merger_builder = merger_builder
        self.dispatcher = dispatcher
        self.classifier = classifier


    def _segment_locate(self, slide_buffer):
        """
        Apply segmentation and localization to the filtered out :class:`Tile`s
        originating from the :class:`TileStream` contained in the given
        :class:`SlideBuffer`

        Parameters
        ---------
        slide_buffer : :class:`SlideBuffer`
            The :class:`SlideBuffer` instance on which the segmentation and
            localization must be applied

        Return
        ------
        dict_polygons : a dictionary of polygons
                    (:class:`shapely.Polygon`)
            key = image_id => sequence of polygons
            The polygons are expressed in "real" coordinates
            (from the bottom left corner of the enclosing image)
        """
        dict_polygons = {}
        for tile_stream in monitor_with("CM.SLDC.SL_slide", task_name="Segment and locate (slides)")(slide_buffer):
            #Get the merger in place
            merger = self.merger_builder.create_merger()
            for tile in monitor_with("CM.SLDC.SL_tile", task_name="Segment and locate (tiles)",)(tile_stream):
                polygons = []
                #Filtering
                if self.filter.filter_tile(tile):
                    #Segmenting
                    segmented = self.segmenter.segment(tile.patch_as_numpy())

                    #Extracting the polygons
                    polygons = self.locator.vectorize(segmented,
                                                      offset=(tile.row_offset,
                                                              tile.col_offset))
                #Storing the polygons (need to call store even if polygons is empty)
                merger.store(tile, polygons)
                print tile
                for p in polygons:
                    print p

            #Merging the polygons from different tiles
            with monitor_with("C_CM.SLDC.merge", task_name="Merging polygons"):
                dict_polygons[tile_stream.get_image_id()] = merger.merge()

#            IPython.embed()
        return dict_polygons

    def segment_locate(self, datastore):
        """
        Apply segmentation and localization to the :class:`SlideBuffer`
        contained in the given :class:`DataStore`

        Parameters
        ---------
        datastore : :class:`DataStore`
            The :class:`DataStore`containing the :class:`SlideBuffer` to
            process

        Return
        ------
        dict_polygons : a dictionary of polygons
                    (:class:`shapely.Polygon`)
            key = image_id => sequence of polygons
            The polygons are expressed in "real" coordinates
            (from the bottom left corner of the enclosing image)
        """
        # Getting the data
        slide_buffer = datastore.get_main_slide_stream()
        # Launching the "process & locate" subroutine
        dict_poly_res = self.submit_to_executor(self._segment_locate, slide_buffer)
        # Flattening
        dict_polygons = {}
        map(dict_polygons.update, dict_poly_res)
        return dict_polygons

    def dispatch_classify(self, datastore):
        """
        Apply the dispatch and classify subroutine to the given
        :class:`DataStore`

        Parameters
        ---------
        datastore : :class:`DataStore`
            The :class:`DataStore` to process

        Return
        ------
        result : classifier dependent
            The result of the classification
        """
        with monitor_with("C_CM.SLDC.dispatch", task_name="Dispatching"):
            self.dispatcher.dispatch(datastore)
        with monitor_with("C_CM.SLDC.classify", task_name="Classifying"):
            res = self.classifier.classify(datastore)
        return res

    def process(self, datastore):
        """
        Apply the whole SLDC routine to the given datastore

        Parameters
        ---------
        datastore : :class:`DataStore`
            The :class:`DataStore`containing the :class:`SlideBuffer` to
            process

        Return
        ------
        result : classifier dependent
            The result of the classification
        """
        dict_polygons = self.segment_locate(datastore)
        datastore.store_polygons(dict_polygons)
        return self.dispatch_classify(datastore)








