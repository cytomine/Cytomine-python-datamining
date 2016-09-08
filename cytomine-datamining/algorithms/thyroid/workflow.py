# -*- coding: utf-8 -*-
import optparse

import os

import time
from PIL.Image import fromarray
from cytomine import Cytomine
from cytomine_utilities.reader import CytomineReader, Bounds
from cytomine_utilities.wholeslide import WholeSlide
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from sldc import Logger, StandardOutputLogger, SilentLogger, Loggable, WorkflowBuilder, WorkflowChainBuilder, \
    batch_split, TileTopology, DefaultTileBuilder

from helpers.datamining.colordeconvoluter import ColorDeconvoluter
from helpers.utilities.datatype.polygon import affine_transform

from segmenters import AggregateSegmenter, SlideSegmenter, get_standard_kernel, get_standard_struct_elem
from dispatching_rules import CellRule, AggregateRule, CellGeometricRule
from image_adapter import CytomineTileBuilder, TileCache, CytomineSlide
from ontology import ThyroidOntology
from classifiers import PyxitClassifierAdapter
from cytomine.models import AlgoAnnotationTerm
from helpers.utilities.cytominejob import CytomineJob


__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def _upload_annotation(cytomine, img_inst, polygon, label=None, proba=1.0):
    image_id = img_inst.id
    # Transform in cartesian coordinates
    polygon = affine_transform(xx_coef=1, xy_coef=0, yx_coef=0, yy_coef=-1, delta_y=img_inst.height)(polygon)

    annotation = cytomine.add_annotation(polygon.wkt, image_id)
    if label is not None and annotation is not None:
        cytomine.add_annotation_term(annotation.id, label, label, proba, annotation_term_model=AlgoAnnotationTerm)


def _first_submit(image, cytomine, results_batch):
    """Submit polygons from the first pass"""
    for polygon, dispatch, cls, proba in results_batch:
        if dispatch == "cell":  # cell
            _upload_annotation(cytomine, image.image_instance, polygon,
                               ThyroidOntology.CELL_INCL if cls == 1 else ThyroidOntology.CELL_NORM, proba)
        elif dispatch == "aggregate":  # pattern
            _upload_annotation(cytomine, image.image_instance, polygon,
                               ThyroidOntology.PATTERN_PROLIF if cls == 1 else ThyroidOntology.PATTERN_NORM, proba)


def _second_submit(image, cytomine, results_batch):
    """Submit polygons from the second pass"""
    for polygon, dispatch, cls, proba in results_batch:
        if dispatch == "cell":
            _upload_annotation(cytomine, image.image_instance, polygon,
                               ThyroidOntology.CELL_INCL if cls == 1 else ThyroidOntology.CELL_NORM, proba)


class ThyroidPostProcessor(Loggable):
    """
    This post processor publish the annotation and classes with the predicted class
    """

    def __init__(self, cytomine, logger=SilentLogger(), n_jobs=1):
        """Build a cytomine post processor
        Parameters
        ----------
        cytomine: Cytomine
            Cytomine client
        logger: Logger (optional, default: a SilentLogger object)
            A logger object
        n_jobs: int (optional, default: 1)
            The number of available processes for sending the results to the server
        """
        Loggable.__init__(self, logger=logger)
        self._cytomine = cytomine
        self._pool = Parallel(n_jobs=n_jobs)

    def post_process(self, image, chain_info):
        """
        Parameters
        ----------
        image: CytomineSlide
        workflow_info: WorkflowInformation

        """
        # if len(workflow_info_collection) != 2:
        #     raise RuntimeError("Two executions expected, got {}.".format(len(workflow_info_collection)))

        # extract polygons from first run
        slide_processing = chain_info.information("slide_processing")
        slide_processing_batches = batch_split(self._pool.n_jobs, slide_processing)
        self._cytomine._Cytomine__conn = None
        self.logger.info("ThyroidPostProcessor: submit results for first pass")
        self._pool(delayed(_first_submit)(image, self._cytomine, batch) for batch in slide_processing_batches)
        self.logger.info("ThyroidPostProcessor: first pass")
        slide_processing.timing.report(self.logger)

        # extract polygons from second run
        aggre_processing = chain_info.information("aggregate_processing")
        aggre_processing_batches = batch_split(self._pool.n_jobs, aggre_processing)
        self._cytomine._Cytomine__conn = None
        self.logger.info("ThyroidPostProcessor: submit results for second pass")
        self._pool(delayed(_first_submit)(image, self._cytomine, batch) for batch in aggre_processing_batches)
        self.logger.info("ThyroidPostProcessor: second pass")
        aggre_processing.timing.report(self.logger)


# helpers for tile caching
def _parallel_cache_tiles(tiles, cytomine, whole_slide, working_path):
    reader = CytomineReader(cytomine, whole_slide)
    for tile in tiles:
        reader.window_position = Bounds(tile.offset_x, tile.offset_y, tile.width, tile.height)
        path = "{}_{}_{}_{}_{}.png".format(whole_slide.image.id, tile.offset_x, tile.abs_offset_y, tile.width, tile.height)
        path = os.path.join(working_path, path)
        if not os.path.exists(path):
            reader.read()
            reader.data.save(path)


class ThyroidJob(CytomineJob, Loggable):
    def __init__(self, cell_classif, aggr_classif, dispatch_classif, host, public_key, private_key, software_id,
                 project_id, slide_ids, working_path="/tmp", protocol="http://", base_path="/api/",
                 verbose=Logger.INFO, timeout=120, n_jobs=1, tile_max_width=1024, tile_max_height=1024, overlap=7):
        """
        Create a standard thyroid application with the given parameters.
        Standard implies :
            - The data source is Cytomine
            - For the first segmentation :
                - Filtering out tiles is done a by :class:`StdFilter`
                - The segmentation is done by a :class:`ColorDeconvoluter`
                - The location is done by a :class:`CV2Locator`
                - The merging of neighboring tiles is done by a
                    :class:`RowOrderMerger`
            - The first dispatching is done by an :class:`AreaDispatcher`
            - The second dispatching is done by an :class:`AreaDispatcher`

        Parameters
        ----------
        cell_classif: string
            The cell classifier pickle file path
        aggr_classif: string
            The architectural pattern classifier pickle file path
        dispatch_classif: string
            The classifier for dispatching cells pickle file path
        host : str
            The Cytomine server host URL
        public_key : str
            The user public key
        private_key : str
            The user corresponding private key
        software_id : int
            The identifier of the software on the Cytomine server
        project_id : int
            The identifier of the project to process on the Cytomine server
        slide_ids : sequence of postive int
            The identifiers of the slides to process
        working_path : str (default : "/tmp")
            A directory for caching temporary files
        protocol : str (default : "http://")
            The communication protocol
        base_path : str (default : /api/)
            n/a
        timeout : int (default : 120)
            The timeout time (in seconds)
        n_jobs : int (!=0) (Default : 1)
            The number of core to use.
                If >0 : the parallelization factor.
                If <0 : nb_tasks = #cpu+nb_cores+1
                Set to -1 to use the maximum number of core
        tile_max_width: int
            The tiles max width
        tile_max_height: int
            The tiles max height
        overlap: int
            Overlap between tiles
        """
        # Create Cytomine instance
        random_state = check_random_state(0)
        Loggable.__init__(self, logger=StandardOutputLogger(verbose))
        cytomine = Cytomine(host, public_key, private_key, working_path, protocol, base_path, verbose >= Logger.DEBUG, timeout)
        CytomineJob.__init__(self, cytomine, software_id, project_id)

        # Build tile builders
        tile_builder = CytomineTileBuilder(cytomine, working_path)
        tile_cache = TileCache(tile_builder, working_path)

        self.logger.i("ThyroidJob: start caching tiles.")
        cache_start = time.time()
        self.cache_tiles(cytomine, slide_ids, tile_max_width=tile_max_width, tile_max_height=tile_max_height,  working_path=working_path, overlap=overlap, n_jobs=n_jobs)
        cache_end = time.time()
        self.logger.i("ThyroidJob: end caching tiles in {} s.".format(cache_end - cache_start))

        # Build useful classifiers and rules
        adapter_builder_func = PyxitClassifierAdapter.build_from_pickle
        aggr_classifier = adapter_builder_func(aggr_classif, tile_cache, self.logger, random_state=random_state, n_jobs=n_jobs)
        cell_classifier = adapter_builder_func(cell_classif, tile_cache, self.logger, random_state=random_state, n_jobs=n_jobs)
        dispatch_classifier = adapter_builder_func(dispatch_classif, tile_cache, self.logger, random_state=random_state, n_jobs=n_jobs)
        cell_rule = CellRule(dispatch_classifier, logger=self.logger)
        aggregate_rule = AggregateRule(dispatch_classifier, logger=self.logger)

        # Build workflows
        workflow_builder = WorkflowBuilder()
        color_deconvoluter = ColorDeconvoluter()
        color_deconvoluter.set_kernel(get_standard_kernel())

        # Builder workflow 1 (slide processing)
        workflow_builder.set_n_jobs(n_jobs)
        workflow_builder.set_parallel_dc(True)
        workflow_builder.set_segmenter(SlideSegmenter(color_deconvoluter))
        workflow_builder.add_classifier(cell_rule, cell_classifier, dispatching_label="cell")
        workflow_builder.add_classifier(aggregate_rule, aggr_classifier, dispatching_label="aggregate")
        workflow_builder.set_tile_size(tile_max_width, tile_max_height)
        workflow_builder.set_overlap(overlap)
        workflow_builder.set_logger(self.logger)
        workflow_builder.set_tile_builder(tile_builder)

        slide_workflow = workflow_builder.get()

        # Build workflow 2 (aggregate processing)
        workflow_builder.set_n_jobs(1)
        workflow_builder.set_segmenter(AggregateSegmenter(color_deconvoluter, struct_elem=get_standard_struct_elem()))
        workflow_builder.add_classifier(CellGeometricRule(), cell_classifier, dispatching_label="cell")
        workflow_builder.set_tile_size(tile_max_width, tile_max_height)
        workflow_builder.set_overlap(overlap)
        workflow_builder.set_logger(self.logger)
        workflow_builder.set_tile_builder(tile_builder)

        aggregate_workflow = workflow_builder.get()

        # Build workflow chain
        chain_builder = WorkflowChainBuilder()
        chain_builder.set_first_workflow(slide_workflow, label="slide_processing")
        chain_builder.add_executor(aggregate_workflow, label="aggregate_processing", logger=self.logger, n_jobs=n_jobs)
        chain_builder.set_logger(self.logger)

        self._chain = chain_builder.get()
        self._slides = [CytomineSlide(cytomine, slide_id) for slide_id in slide_ids]
        self._post_processor = ThyroidPostProcessor(cytomine, logger=self.logger, n_jobs=n_jobs)

    def cache_tiles(self, cytomine, slides, tile_max_width=1024, tile_max_height=1024, overlap=7,
                    working_path="/tmp/sldc/", n_jobs=1):
        """"""
        if not os.path.exists(working_path):
            os.makedirs(working_path)

        pool = Parallel(n_jobs=min(6, n_jobs))
        for id in slides:
            slide = CytomineSlide(cytomine, id)
            topology = TileTopology(slide, DefaultTileBuilder(), max_width=tile_max_width,
                                    max_height=tile_max_height, overlap=overlap)
            whole_slide = WholeSlide(slide.image_instance)
            tile_batches = batch_split(n_jobs, topology)
            cytomine._Cytomine__conn = None
            pool(delayed(_parallel_cache_tiles)(tiles, cytomine, whole_slide, working_path) for tiles in tile_batches)

    def run(self):
        # try:
        self.logger.info("ThyroidJob : start thyroid workflow.")
        for slide in self._slides:
            results = self._chain.process(slide)
            self._post_processor.post_process(slide, results)
        self.logger.info("ThyroidJob : end thyroid workflow.")
        # except Exception as e:
        #     self.logger.error("ThyroidJob : error while executing workflow \n" +
        #                       "ThyroidJob : \"{}\".".format(e.message))


def arr2str(arr):
    return "".join(str(arr).strip("[]").split(","))


def str2list(l, conv=int):
    return [conv(v) for v in l.split(",")]


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main(argv):
    # ----------Parsing command line args----------- #
    parser = optparse.OptionParser()  # TODO desc.
    parser.add_option("--cell_classifier", type='string', default="", dest="cell_classifier",
                      help="File where the cell classifier has been pickled")
    parser.add_option("--aggregate_classifier", type='string', default="", dest="aggregate_classifier",
                      help="File where the architectural pattern classifier has been pickled")
    parser.add_option("--dispatch_classifier", type='string', default="", dest="dispatch_classifier",
                      help="File where the dispatch classifier has been pickled")
    parser.add_option("--host", type='string', default="", dest="host",
                      help="Cytomine server host URL")
    parser.add_option("--public_key", type='string', default="", dest="public_key",
                      help="User public key")
    parser.add_option("--private_key", type='string', default="", dest="private_key",
                      help="User Private key")
    parser.add_option("--software_id", type='int', dest="software_id",
                      help="Identifier of the software on the Cytomine server")
    parser.add_option("--project_id", type='int', dest="project_id",
                      help="Identifier of the project to process on the Cytomine server")
    parser.add_option("--slide_ids", type='string', default="", dest="slide_ids",
                      help="Sequence of ids of the slides to process")
    parser.add_option("--working_path", type='string', default="/tmp", dest="working_path",
                      help="Directory for caching temporary files")
    parser.add_option("--protocol", type='string', default="http://", dest="protocol",
                      help="Communication protocol")
    parser.add_option("--base_path", type='string', default="/api/", dest="base_path",
                      help="Base path for api url")
    parser.add_option("--timeout", type='int', default=120, dest="timeout",
                      help="Timeout time for connection (in seconds)")
    parser.add_option("--verbose", type='int', default=Logger.INFO, dest="verbose",
                      help="Output verbosity in {0,1,2,3,4}")
    parser.add_option("--n_jobs", type='int', default=1, dest="n_jobs",
                      help="Number of core to use")
    parser.add_option("--tile_max_width", type='int', default=1024, dest="tile_max_width",
                      help="Slides max width")
    parser.add_option("--tile_max_height", type='int', default=1024, dest="tile_max_height",
                      help="Slides max height")

    options, arguments = parser.parse_args(args=argv)

    with ThyroidJob(options.cell_classifier, options.aggregate_classifier,
                    options.dispatch_classifier, options.host, options.public_key, options.private_key,
                    options.software_id, options.project_id, str2list(options.slide_ids),
                    verbose=options.verbose, n_jobs=options.n_jobs, protocol=options.protocol,
                    base_path=options.base_path, working_path=options.working_path,
                    tile_max_height=options.tile_max_height, tile_max_width=options.tile_max_width) as job:
        job.run()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
