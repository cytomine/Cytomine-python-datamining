# -*- coding: utf-8 -*-
import argparse

from sldc import PostProcessor, WorkflowChain
from helpers.utilities.datatype.polygon import affine_transform
from image_providers import SlideProvider
from slide_processing import SlideProcessingWorkflow
from image_adapter import CytomineTileBuilder, TileCache
from classifiers import PyxitClassifierAdapter
from cytomine import Cytomine
from cytomine.models import AlgoAnnotationTerm
from helpers.utilities.argparsing import positive_int, positive_float
from helpers.utilities.argparsing import not_zero, range0_255
from helpers.utilities.cytominejob import CytomineJob


__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class CytominePostProcessor(PostProcessor):
    """
    This post processor publish the annotation and classes with the predicted class
    """

    def __init__(self, cytomine):
        self._cytomine = cytomine

    def post_process(self, image, polygons_classes):
        for polygon, cls in polygons_classes:
            if cls is not None:
                self._upload_annotation(image.image_instance, polygon, int(cls))

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
                 nb_jobs=1, *args, **kwargs):
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
        nb_jobs : int (!=0) (Default : 1)
            The number of core to use.
                If >0 : the parallelization factor.
                If <0 : nb_tasks = #cpu+nb_cores+1
                Set to -1 to use the maximum number of core
        zoom_sl : int >= 0 (default : 1)
            The zoom level for the segment-locate part
        tile_filter_min_std : float (default : 10.)
            The minimum standard deviation needed by a tile to be treated
        deconv_kernel : 3x3 float np.ndarray (default :
        :meth:`get_standard_kernel`)
            The kernel for the color deconvolution
        seg_threshold : int in [0, 255] (default : 120)
            The threshold used in the first segmentation
        seg_struct_elem : binary 2D np.ndarray
                         (default : :meth:`get_standard_struct_elem`)
            The structural element for the morphological operations
        seg_nb_morph_iter : a list of 3 int (default : [1, 3, 7])
            The sequence of morphological operations. see :class:`CDSegmenter`
        merg_boundary_thickness : int (default : 2)
            The boundary thickness. That is, the distance at which an object
            is considered as touching the boundary. See :class:`Merger` for
            more information
        disp1_cell_min_area : float
            The cells minimum area for the first dispatching. It must be
            consistent with the polygon coordinate system. In particular
            with the scale
        disp1_cell_max_area : float
            The cells maximum area for the first dispatching. It must be
            consistent with the polygon coordinate system. In particular
            with the scale
        disp1_cell_min_circ : float
            The cells minimum circularity for the first dispatching. It
            must be consistent with the polygon coordinate system. In
            particular with the scale
        disp1_clust_min_cell_nb : int
            The minimum number of cells to form a cluster for the first
            dispatching. It must be consistent with the polygon coordinate
            system. In particular with the scale
        disp2_cell_min_area : float
            The cells minimum area for the second dispatching. It must be
            consistent with the polygon coordinate system. In particular
            with the scale
        disp2_cell_max_area : float
            The cells maximum area for the second dispatching. It must be
            consistent with the polygon coordinate system. In particular
            with the scale
        disp2_cell_min_circ : float
            The cells minimum circularity for the second dispatching. It must
            be consistent with the polygon coordinate system. In particular
            with the scale
        disp2_clust_min_cell_nb : int
            The minimum number of cells to form a cluster for the second
            dispatching. It must be consistent with the polygon coordinate
            system. In particular with the scale
        """
        # Create Cytomine instance
        tile_max_width, tile_max_height = 1024, 1024
        cytomine = Cytomine(host, public_key, private_key, working_path, protocol, base_path, verbose, timeout)
        CytomineJob.__init__(self, cytomine, software_id, project_id)
        tile_builder = CytomineTileBuilder(cytomine)
        tile_cache = TileCache(tile_builder)
        aggr_classifier = PyxitClassifierAdapter.build_from_pickle(aggregate_classifier, tile_cache, working_path)
        cell_classifier = PyxitClassifierAdapter.build_from_pickle(cell_classifier, tile_cache, working_path)
        aggr_dispatch = PyxitClassifierAdapter.build_from_pickle(aggregate_dispatch_classifier, tile_cache, working_path)
        cell_dispatch = PyxitClassifierAdapter.build_from_pickle(cell_dispatch_classifier, tile_cache, working_path)
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


def main(argv):
    print argv  # TODO remove
    # ----------Parsing command line args----------- #
    parser = argparse.ArgumentParser()  # TODO desc.
    parser.add_argument("cell_classifier",           help="File where the cell classifier has been pickled")
    parser.add_argument("aggregate_classifier",      help="File where the architectural pattern classifier has been pickled")
    parser.add_argument("cell_dispatch_classifier",  help="File where the cell dispatch classifier has been pickled")
    parser.add_argument("aggregate_dispatch_classifier", help="File where the aggregate dispatch classifier has been pickled")
    parser.add_argument("host",                      help="Cytomine server host URL")
    parser.add_argument("public_key",                help="User public key")
    parser.add_argument("private_key",               help="User Private key")
    parser.add_argument("software_id",               help="Identifier of the software on the Cytomine server")
    parser.add_argument("project_id",                help="Identifier of the project to process on the Cytomine server")
    parser.add_argument("slide_ids",                 help="Sequence of ids of the slides to process", nargs="+", type=int)
    parser.add_argument("--working_path",            help="Directory for caching temporary files", default="/tmp")
    parser.add_argument("--protocol",                help="Communication protocol",default="http://")
    parser.add_argument("--base_path",               help="n/a", default="/api/")
    parser.add_argument("--timeout",                 help="Timeout time for connection (in seconds)", type=positive_int, default="120")
    parser.add_argument("--verbose",                 help="increase output verbosity", action="store_true", default=True)
    parser.add_argument("--nb_jobs",                 help="Number of core to use", type=not_zero, default=1)
    # parser.add_argument("--disp1_cell_min_area",     help="Cell minimum area for the first dispatching", type=positive_float, default=600)
    # parser.add_argument("--disp1_cell_max_area",     help="Cell maximum area for the first dispatching", type=positive_float, default=2000)
    # parser.add_argument("--disp1_cell_min_circ",     help="Cell minimum circularity for the first dispatching", type=positive_float, default=.70)
    # parser.add_argument("--disp1_clust_min_cell_nb", help="Minimum number of cells in a cluster for the first dispatching", type=positive_int, default=3)
    # parser.add_argument("--disp2_cell_min_area",     help="Cell minimum area for the second dispatching", type=positive_float, default=800)
    # parser.add_argument("--disp2_cell_max_area",     help="Cell maximum area for the second dispatching", type=positive_float, default=4000)
    # parser.add_argument("--disp2_cell_min_circ",     help="Cell minimum circularity for the second dispatching", type=positive_float, default=.85)
    # parser.add_argument("--disp2_clust_min_cell_nb", help="Minimum number of cells in a cluster for the second dispatching", type=positive_float, default=1)

    args = parser.parse_args(args=argv)

    with ThyroidJob(args.cell_classifier, args.aggregate_classifier, args.cell_dispatch_classifier,
                    args.aggregate_dispatch_classifier, args.host, args.public_key, args.private_key, args.software_id,
                    args.project_id, args.slide_ids) as job:
        job.run()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
