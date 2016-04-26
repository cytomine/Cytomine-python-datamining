# -*- coding: utf-8 -*-
import optparse

from sldc import PostProcessor, WorkflowChain, FullImageWorkflowExecutor, Logger, StandardOutputLogger, SilentLogger, \
    Loggable
from helpers.utilities.datatype.polygon import affine_transform
from image_providers import SlideProvider, AggregateWorkflowExecutor
from slide_processing import SlideProcessingWorkflow
from image_adapter import CytomineTileBuilder, TileCache, CytomineMaskedTileBuilder
from aggregate_processing import AggregateProcessingWorkflow
from ontology import ThyroidOntology
from classifiers import PyxitClassifierAdapter
from cytomine import Cytomine
from cytomine.models import AlgoAnnotationTerm
from helpers.utilities.cytominejob import CytomineJob


__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class CytominePostProcessor(PostProcessor):
    """
    This post processor publish the annotation and classes with the predicted class
    """

    def __init__(self, cytomine, logger=SilentLogger()):
        """Build a cytomine post processor
        Parameters
        ----------
        cytomine: Cytomine
            Cytomine client
        logger: Logger (optional, default: a SilentLogger object)
            A logger object
        """
        PostProcessor.__init__(self, logger=logger)
        self._cytomine = cytomine

    def post_process(self, image, workflow_info_collection):
        # if len(workflow_info_collection) != 2:
        #     raise RuntimeError("Two executions expected, got {}.".format(len(workflow_info_collection)))

        # extract polygons from first run
        slide_processing = workflow_info_collection[0]
        for polygon, dispatch, cls in slide_processing.iterator():
            upload_fn = self._upload_fn(image, polygon)
            if dispatch == 0:  # cell
                upload_fn(ThyroidOntology.CELL_INCL if cls == 1 else ThyroidOntology.CELL_NORM)
            elif dispatch == 1:  # pattern
                upload_fn(ThyroidOntology.PATTERN_PROLIF if cls == 1 else ThyroidOntology.PATTERN_NORM)
        slide_processing.timing.report(self.logger)

        # # extract polygons from second run
        # aggre_processing = workflow_info_collection[1]
        # for polygon, dispatch, cls in aggre_processing.iterator():
        #     upload_fn = self._upload_fn(image, polygon)
        #     if dispatch == 0:
        #         upload_fn(ThyroidOntology.CELL_INCL if cls == 1 else ThyroidOntology.CELL_NORM)

    def _upload_fn(self, image, polygon):
        """Return a callable taking one parameter. This callable uploads the polygon as annotation for the given image.
        The callable parameter is the label to associate to the annotation
        """
        return lambda cls: self._upload_annotation(image.image_instance, polygon, label=cls)

    def _upload_annotation(self, img_inst, polygon, label=None):
        image_id = img_inst.id
        # Transform in cartesian coordinates
        polygon = affine_transform(xx_coef=1, xy_coef=0, yx_coef=0, yy_coef=-1, delta_y=img_inst.height)(polygon)

        annotation = self._cytomine.add_annotation(polygon.wkt, image_id)
        if label is not None and annotation is not None:
            self._cytomine.add_annotation_term(annotation.id, label, label, 1.0, annotation_term_model=AlgoAnnotationTerm)


class ThyroidJob(CytomineJob, Loggable):
    def __init__(self, cell_classif, aggr_classif, cell_dispatch_classif, aggr_dispatch_classif,
                 host, public_key, private_key, software_id, project_id,
                 slide_ids, working_path="/tmp", protocol="http://", base_path="/api/", verbose=Logger.INFO,
                 timeout=120, n_jobs=1, tile_max_width=1024, tile_max_height=1024):
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
        cell_dispatch_classif: string
            The classifier for dispatching cells pickle file path
        aggr_dispatch_classif: string
            The classifier for dispaching aggregates pickle file path
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
        """
        # Create Cytomine instance
        Loggable.__init__(self, logger=StandardOutputLogger(verbose))
        cytomine = Cytomine(host, public_key, private_key, working_path, protocol,
                            base_path, verbose >= Logger.DEBUG, timeout)
        CytomineJob.__init__(self, cytomine, software_id, project_id)
        tile_builder = CytomineTileBuilder(cytomine)
        masked_tile_builder = CytomineMaskedTileBuilder(cytomine)
        tile_cache = TileCache(tile_builder, working_path)

        # build useful classifiers
        adapter_builder_func = PyxitClassifierAdapter.build_from_pickle
        aggr_classifier = adapter_builder_func(aggr_classif, tile_cache, self._logger, n_jobs=n_jobs)
        cell_classifier = adapter_builder_func(cell_classif, tile_cache, self._logger, n_jobs=n_jobs)
        aggr_dispatch = adapter_builder_func(aggr_dispatch_classif, tile_cache, self._logger, n_jobs=n_jobs)
        cell_dispatch = adapter_builder_func(cell_dispatch_classif, tile_cache, self._logger, n_jobs=n_jobs)

        # build workflow chain components
        image_provider = SlideProvider(cytomine, slide_ids)
        slide_workflow = SlideProcessingWorkflow(tile_builder, cell_classifier, aggr_classifier, cell_dispatch,
                                                 aggr_dispatch, tile_max_width=tile_max_width,
                                                 tile_max_height=tile_max_height, logger=self._logger)
        aggre_workflow = AggregateProcessingWorkflow(masked_tile_builder, cell_classifier, cell_dispatch,
                                                     tile_max_width=tile_max_width, tile_max_height=tile_max_height,
                                                     logger=self._logger)
        aggre_workflow_executor = AggregateWorkflowExecutor(cytomine, aggre_workflow, logger=self._logger)
        post_processor = CytominePostProcessor(cytomine, logger=self._logger)

        # build the workflow chain
        workflow_executors = [FullImageWorkflowExecutor(slide_workflow)]  # , aggre_workflow_executor]
        self._chain = WorkflowChain(image_provider, workflow_executors, post_processor, logger=self._logger)

    def run(self):
        try:
            self._logger.info("ThyroidJob : start thyroid workflow.")
            self._chain.execute()
            self._logger.info("ThyroidJob : end thyroid workflow.")
        except Exception as e:
            self._logger.warning("ThyroidJob : error while executing workflow : {}.".format(e.message))


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
    parser.add_option("--cell_dispatch_classifier", type='string', default="", dest="cell_dispatch_classifier",
                      help="File where the cell dispatch classifier has been pickled")
    parser.add_option("--aggregate_dispatch_classifier", type='string', default="", dest="aggregate_dispatch_classifier",
                      help="File where the aggregate dispatch classifier has been pickled")
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

    with ThyroidJob(options.cell_classifier, options.aggregate_classifier, options.cell_dispatch_classifier,
                    options.aggregate_dispatch_classifier, options.host, options.public_key, options.private_key,
                    options.software_id, options.project_id, str2list(options.slide_ids),
                    verbose=options.verbose, n_jobs=options.n_jobs, protocol=options.protocol,
                    base_path=options.base_path, working_path=options.working_path,
                    tile_max_height=options.tile_max_height, tile_max_width=options.tile_max_width) as job:
        job.run()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
