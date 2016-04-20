# -*- coding: utf-8 -*-
import optparse

from sldc import PostProcessor, WorkflowChain
from helpers.utilities.datatype.polygon import affine_transform
from image_providers import SlideProvider
from slide_processing import SlideProcessingWorkflow
from image_adapter import CytomineTileBuilder, TileCache
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

    def __init__(self, cytomine):
        self._cytomine = cytomine

    def post_process(self, image, workflow_info_collection):
        for polygon, dispatch, cls in workflow_info_collection.polygons():
            upload_fn = self._upload_fn(image, polygon)
            if dispatch == 0:  # cell
                upload_fn(ThyroidOntology.CELL_INCL if cls == 1 else ThyroidOntology.CELL_NORM)
            elif dispatch == 1:  # pattern
                upload_fn(ThyroidOntology.PATTERN_PROLIF if cls == 1 else ThyroidOntology.PATTERN_NORM)

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


class ThyroidJob(CytomineJob):

    def __init__(self, cell_classifier, aggregate_classifier, cell_dispatch_classifier, aggregate_dispatch_classifier,
                 host, public_key, private_key, software_id, project_id,
                 slide_ids, working_path="/tmp", protocol="http://", base_path="/api/", verbose=False, timeout=120,
                 n_jobs=1, tile_max_width=1024, tile_max_height=1024):
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
        cell_classifier : `callable`
            The cell classifier
        pattern_classifier : `callable`
            The architectural pattern classifier
        cell_dispatch_classifier: PyxitClassifierAdapter
            The classifier for dispatching cells
        aggregate_dispatch_classifier: PyxitClassifierAdapater
            The classifier for dispaching aggregates
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
        cytomine = Cytomine(host, public_key, private_key, working_path, protocol, base_path, verbose, timeout)
        CytomineJob.__init__(self, cytomine, software_id, project_id)
        tile_builder = CytomineTileBuilder(cytomine)
        tile_cache = TileCache(tile_builder)
        aggr_classifier = PyxitClassifierAdapter.build_from_pickle(aggregate_classifier, tile_cache, working_path,
                                                                   n_jobs=n_jobs, verbose=verbose)
        cell_classifier = PyxitClassifierAdapter.build_from_pickle(cell_classifier, tile_cache, working_path,
                                                                   n_jobs=n_jobs, verbose=verbose)
        aggr_dispatch = PyxitClassifierAdapter.build_from_pickle(aggregate_dispatch_classifier, tile_cache,
                                                                 working_path, n_jobs=n_jobs, verbose=verbose)
        cell_dispatch = PyxitClassifierAdapter.build_from_pickle(cell_dispatch_classifier, tile_cache, working_path,
                                                                 n_jobs=n_jobs, verbose=verbose)
        image_provider = SlideProvider(cytomine, slide_ids)
        slide_workflow = SlideProcessingWorkflow(tile_builder, cell_classifier, aggr_classifier, cell_dispatch,
                                                 aggr_dispatch, tile_max_width=tile_max_width,
                                                 tile_max_height=tile_max_height)
        post_processor = CytominePostProcessor(cytomine)
        self._chain = WorkflowChain(image_provider, slide_workflow, post_processor)

    def run(self):
        self._chain.execute()


def arr2str(arr):
    return "".join(str(arr).strip("[]").split(","))


def str2list(l, conv=int):
    print "PRINT: {}".format(l)
    return [conv(v) for v in l.split(",")]


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main(argv):
    print argv  # TODO remove
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
    parser.add_option("--verbose", type='string', default=True, dest="verbose",
                      help="increase output verbosity")
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
                    verbose=str2bool(options.verbose), n_jobs=options.n_jobs,
                    tile_max_height=options.tile_max_height, tile_max_width=options.tile_max_width) as job:
        job.run()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
