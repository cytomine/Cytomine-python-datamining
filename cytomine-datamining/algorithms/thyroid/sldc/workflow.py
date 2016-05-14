# -*- coding: utf-8 -*-
from joblib import Parallel, delayed

from image import Image, TileBuilder, TileTopologyIterator
from merger import Merger
from locator import Locator
from information import WorkflowInformation
from logging import Loggable, SilentLogger
from timing import WorkflowTiming
from errors import TileExtractionException

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


def _parallel_sl_with_timing(tile, segmenter, locator):
    """Helper function for parallel execution. Error occurring in this method is notified by returning None in place of
    the timing object.

    Parameters
    ----------
    tile: Tile
        The tile object to process
    segmenter: Segementer
        The segmenter object
    locator: Locator
        The locator object

    Returns
    -------
    timing: WorkflowTiming
        The timing of execution for processing of the tile.
    tile: Tile
        The tile object
    polygons: iterable (subtype: shapely.geometry.Polygon)
        The polygons found in the tile
    """
    timing = WorkflowTiming()
    timing.start_fetching()
    np_image = tile.np_image
    timing.end_fetching()
    timing.start_segment()
    segmented = segmenter.segment(np_image)
    timing.end_segment()
    timing.start_location()
    located = locator.locate(segmented, offset=tile.offset)
    timing.end_location()
    return timing, tile, located


class SLDCWorkflow(Loggable):
    """A class that coordinates various components of the SLDC workflow in order to detect objects and return
    their information.
    """

    def __init__(self, segmenter, dispatcher_classifier, tile_builder,
                 tile_max_width=1024, tile_max_height=1024, tile_overlap=5,
                 boundary_thickness=7, logger=SilentLogger(), worker_pool=None):
        """Constructor for SLDCWorkflow objects

        Parameters
        ----------
        segmenter: Segmenter
            The segmenter implementing segmentation procedure to apply on tiles.
        dispatcher_classifier: DispatcherClassifier
            The dispatcher classifier object for dispatching polygons and classify them.
        tile_builder: TileBuilder
            An object for building specific tiles
        tile_max_width: int (optional, default: 1024)
            The maximum width of the tiles when iterating over the image
        tile_max_height: int (optional, default: 1024)
            The maximum height of the tiles when iterating over the image
        tile_overlap: int (optional, default: 5)
            The number of pixels of overlap between tiles when iterating over the image
        boundary_thickness: int (optional, default, 7)
            The thickness between of the boundaries between the tiles for merging
        logger: Logger (optional, default: SilentLogger)
            A logger object
        worker_pool: Parallel (optional, default: null)
            The number of jobs for segmenting and locating the tiles
        """
        Loggable.__init__(self, logger)
        self._tile_max_width = tile_max_width
        self._tile_max_height = tile_max_height
        self._tile_overlap = tile_overlap
        self._tile_builder = tile_builder
        self._segmenter = segmenter
        self._locator = Locator()
        self._merger = Merger(boundary_thickness)
        self._dispatch_classifier = dispatcher_classifier
        self._pool = worker_pool
        self._n_jobs = worker_pool.n_jobs if self._pool is not None else 1

    def process(self, image):
        """Process the given image using the workflow
        Parameters
        ----------
        image: Image
            The image to process

        Returns
        -------
        workflow_information: WorkflowInformation
            The workflow information object containing all the information about detected objects, execution times...

        Notes
        -----
        This method doesn't modify the image passed as parameter.
        This method doesn't modify the object's attributes.
        """
        timing = WorkflowTiming()
        tile_topology = image.tile_topology(max_width=self._tile_max_width, max_height=self._tile_max_height,
                                            overlap=self._tile_overlap)
        tile_iterator = TileTopologyIterator(self._tile_builder, tile_topology, silent_fail=True)

        # segment locate
        self.logger.info("SLDCWorkflow : start segment/locate.")
        if self._n_jobs == 1:
            polygons_tiles = self._sl_sequential(tile_iterator, tile_topology, timing)
        else:
            polygons_tiles = self._sl_parallel(tile_iterator, timing)

        # log end of segment locate
        self.logger.info("SLDCWorkflow : end segment/locate.\n" +
                         "SLDCWorkflow : {} tile(s) processed in {} s.\n".format(len(polygons_tiles), timing.sl_total_duration()) +
                         "SLDCWorkflow : {} polygon(s) found on those tiles.".format(sum([len(polygons) for _, polygons in polygons_tiles])))

        # merge
        self.logger.info("SLDCWorkflow : start merging")
        timing.start_merging()
        polygons = self._merger.merge(polygons_tiles, tile_topology)
        timing.end_merging()
        self.logger.info("SLDCWorkflow : end merging.\n" +
                         "SLDCWorkflow : {} polygon(s) found.\n".format(len(polygons)) +
                         "SLDCWorkflow : executed in {} s.".format(timing.duration_of(WorkflowTiming.MERGING)))

        # dispatch classify
        self.logger.info("SLDCWorkflow : start dispatch/classify.")
        pred, proba, dispatch_indexes = self._dispatch_classifier.dispatch_classify_batch(image, polygons, timing)
        self.logger.info("SLDCWorkflow : end dispatch/classify.\n" +
                         "SLDCWorkflow : executed in {} s.".format(timing.dc_total_duration()))

        return WorkflowInformation(polygons, dispatch_indexes, pred, proba, timing, metadata=self.get_metadata())

    def _segment_locate(self, tile, timing):
        """Fetch a tile and applies it segmentation and location

        Parameters
        ----------
        tile: Tile
            The tile to process for the segment locate
        timing: WorkflowTiming
            The workflow timing object for measuring the execution times of the various steps

        Returns
        -------
        polygons: iterable (subtype: shapely.geometry.Polygon)
            Iterable containing the polygons found by the locate step
        """
        try:
            timing.start_fetching()
            np_image = tile.np_image
            timing.end_fetching()
            timing.start_segment()
            segmented = self._segmenter.segment(np_image)
            timing.end_segment()
            timing.start_location()
            located = self._locator.locate(segmented, offset=tile.offset)
            timing.end_location()
            return located
        except TileExtractionException as e:
            self.logger.warning("SLDCWorkflow : skip tile {} because it couldn't be extracted.\n".format(tile.identifier) +
                                "SLDCWorkflow : fetch error : {}".format(e.message))
            return []

    def get_metadata(self):
        """Return the metadata associated with this workflow
        The metadata should be a way for the implementor to document the results of the workflow execution.
        Returns
        -------
        metadata: string
            The workflow metadata
        """
        return "Workflow class: {}\n".format(self.__class__.__name__)

    def _sl_sequential(self, tile_iterator, tile_topology, timing):
        """Execute the segment locate phase in a sequential order
        Parameters
        ----------
        tile_iterator: TileIterator
            An initialized tile iterator
        tile_topology: TileTopology
            A tile topology
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles_polygons: iterable
            An iterable containing tuples (tile, polygons) where the tile is a Tile object and polygons another iterable
            containing the polygons found on the tile
        """
        polygons_tiles = list()
        for i, tile in enumerate(tile_iterator):
            # log percentage of progress if there are enough tiles
            if tile_topology.tile_count > 25 and (i + 1) % (tile_topology.tile_count // 10) == 0:
                self.logger.info("SLDCWorkflow : {}/{} tiles processed (segment/locate).\n".format(i+1, tile_topology.tile_count) +
                                 "SLDCWorkflow : segment/locate duration is {} s until now.".format(timing.sl_total_duration()))
            polygons_tiles.append((tile, self._segment_locate(tile, timing)))
        return polygons_tiles

    def _sl_parallel(self, tile_iterator, timing):
        """Execute the segment locate phase in parallel
        Parameters
        ----------
        tile_iterator: TileIterator
            An initialized tile iterator
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles_polygons: iterable
            An iterable containing tuples (tile, polygons) where the tile is a Tile object and polygons another iterable
            containing the polygons found on the tile
        """
        # execute in parallel
        timing.start_fsl()
        results = self._pool(delayed(_parallel_sl_with_timing)(tile, self._segmenter, self._locator)
                             for tile in tile_iterator)
        timing.end_fsl()
        # merge sub timings
        for sub_timing, _, _ in results:
            if sub_timing is None:
                self._logger.warning("SLDCWorkflow : Tile {} couldn't be fetched during parallel computations.")
            timing.merge(sub_timing)
        # return tiles polygons
        return [(tile, [] if sub_timing is None else polygons) for sub_timing, tile, polygons in results]

    def _sl_with_timing(self, tile):
        """Execute fetching, segmentation and location. The timing object is built in this function and passed to
        self._segment_locate.

        Parameters
        ----------
        tile: Tile
            The tile to process

        Returns
        -------
        timing: WorkflowTiming
            The timing object containing information about the execution times for the tile
        tile: Tile
            The processed tile
        polygons: iterable
            The found polygons
        """
        timing = WorkflowTiming()
        polygons = self._segment_locate(tile, timing)
        return timing, tile, polygons
