# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.

Note
----
The location phase alter somewhat the geometry. Better be generous on the
boundary distance for merging
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import argparse

from dispatcher import ThyroidDispatcher, AreaDispatcher
from classifier import ThyroidClassifier
from cytominedatastore import ThyroidCytomineDataStore
# TODO replace accordingly
from dummyclassifier import DummyClassifier



from cytomine import Cytomine
from helpers.workflow import SLDCWorkflow
from helpers.datamining import StdFilter
from helpers.datamining import CDSegmenter
from helpers.datamining import ColorDeconvoluter
from helpers.datamining import MergerFactory, RowOrderMerger
from helpers.datamining import CV2Locator

from helpers.utilities.argparsing import positive_int, positive_float
from helpers.utilities.argparsing import not_zero, range0_255
from helpers.utilities.cytominejob import CytomineJob
from helpers.utilities.taskmanager import SerialExecutor, ParallelExecutor



def get_standard_struct_elem():
    """Return the standard structural element"""
    struct_elem = np.array([[0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0], ],
                           dtype=np.uint8)
    return struct_elem

def get_standard_kernel():
    """Return the standard color deconvolution kernel"""
    kernel = np.array([[56.24850493, 71.98403122, 22.07749587],
                       [48.09104103, 62.02717516, 37.36866958],
                       [9.17867488, 10.89206473, 5.99225756]])
    return kernel

class ThyroidJob(CytomineJob):

    def __init__(self,
                 cell_classifier,
                 pattern_classifier,
                 host,
                 public_key,
                 private_key,
                 software_id,
                 project_id,
                 slide_ids,
                 working_path="/tmp",
                 protocol="http://",
                 base_path="/api/",
                 verbose=False,
                 timeout=120,
                 nb_jobs=1,
                 zoom_sl=1,
                 tile_filter_min_std=10.,
                 deconv_kernel=get_standard_kernel(),
                 seg_threshold=120,
                 seg_struct_elem=get_standard_struct_elem(),
                 seg_nb_morph_iter=[1, 3, 7],
                 merg_boundary_thickness=2,
                 disp1_cell_min_area=600,
                 disp1_cell_max_area=2000,
                 disp1_cell_min_circ=.7,
                 disp1_clust_min_cell_nb=3,
                 disp2_cell_min_area=800,
                 disp2_cell_max_area=4000,
                 disp2_cell_min_circ=.85,
                 disp2_clust_min_cell_nb=1,
                 *args, **kwargs):
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
        cytomine_client = Cytomine(host, public_key, private_key,
                                   working_path, protocol, base_path,
                                   verbose, timeout)

        CytomineJob.__init__(self, cytomine_client, software_id, project_id)

        # Create Datastore
        data_store = ThyroidCytomineDataStore(cytomine_client,
                                              slide_ids,
                                              zoom_sl,
                                              working_path)

        # Create TileFilter
        tile_filter = StdFilter(tile_filter_min_std)

        # Create Segmenter
        deconvoluter = ColorDeconvoluter()
        deconvoluter.set_kernel(deconv_kernel)
        segmenter = CDSegmenter(deconvoluter,
                                seg_threshold,
                                seg_struct_elem,
                                seg_nb_morph_iter)

        # Create Locator
        locator = CV2Locator()

        # Create MergerBuilder
        merger_builder = MergerFactory(merg_boundary_thickness, RowOrderMerger)

        # Create Dispatcher
        dispatch_algo = AreaDispatcher(disp1_cell_min_area,
                                       disp1_cell_max_area,
                                       disp1_cell_min_circ,
                                       disp1_clust_min_cell_nb)

        dispatcher_algo2 = AreaDispatcher(disp2_cell_min_area,
                                          disp2_cell_max_area,
                                          disp2_cell_min_circ,
                                          disp2_clust_min_cell_nb)

        dispatcher = ThyroidDispatcher(dispatch_algo,
                                       dispatcher_algo2)
        # Create Classifier
        classifier = ThyroidClassifier(cell_classifier,
                                       pattern_classifier)
        # Create TaskExecutor
        if nb_jobs == 1:
            task_executor = SerialExecutor()
        else:
            task_exec_verbosity = 0
            if verbose:
                task_exec_verbosity = 10
            task_executor = ParallelExecutor(nb_jobs, task_exec_verbosity)
        # Create Miner
        thyroid_miner = SLDCWorkflow(tile_filter, segmenter, locator,
                                     merger_builder, dispatcher, classifier,
                                     task_executor)

        # Set the sl_worflow
        dispatcher.set_sl_workflow(thyroid_miner)

        self._miner = thyroid_miner
        self._store = data_store

    def run(self):
        cell_classif, arch_pattern_classif = self._miner.process(self._store)
        self._store.publish_results(cell_classif, arch_pattern_classif)

def main(argv):
    print argv  #TODO remove
    #----------Parsing command line args-----------#
    parser = argparse.ArgumentParser()  #TODO desc.
    parser.add_argument("cell_classifier",
                        help="File where the cell classifier has been pickled")
    parser.add_argument("pattern_classifier",
                        help="File where the architectural pattern classifier has been pickled")
    parser.add_argument("host",
                        help="Cytomine server host URL")
    parser.add_argument("public_key",
                        help="User public key")
    parser.add_argument("private_key",
                        help="User Private key")
    parser.add_argument("software_id",
                        help="Identifier of the software on the Cytomine server")
    parser.add_argument("project_id",
                        help="Identifier of the project to process on the Cytomine server")
    parser.add_argument("slide_ids",
                        help="Sequence of ids of the slides to process",
                        nargs="+", type=int)
    parser.add_argument("--working_path",
                        help="Directory for caching temporary files",
                        default="/tmp")
    parser.add_argument("--protocol",
                        help="Communication protocol",
                        default="http://")
    parser.add_argument("--base_path",
                        help="n/a",
                        default="/api/")
    parser.add_argument("--timeout",
                        help="Timeout time for connection (in seconds)",
                        type=positive_int,
                        default="120")
    # parser.add_argument("-v" "--verbose", help="increase output verbosity",
    #                 action="store_true")
    parser.add_argument("--nb_jobs",
                        help="Number of core to use",
                        type=not_zero,
                        default=1)
    parser.add_argument("-z", "--zoom_sl",
                        help="Zoom level for the first segmentation",
                        type=positive_int,
                        default=1)
    parser.add_argument("--tile_filter_min_std",
                        help="Minimum standard deviation needed by a tile to be treated",
                        type=positive_float,
                        default=10)
    parser.add_argument("--seg_threshold",
                        help="Threshold used in the first segmentation",
                        type=range0_255,
                        default=120)
    parser.add_argument("--merg_boundary_thickness",
                        help="Boundary thickness for the merging",
                        type=positive_int,
                        default=2)
    parser.add_argument("--disp1_cell_min_area",
                        help="Cell minimum area for the first dispatching",
                        type=positive_float,
                        default=600)
    parser.add_argument("--disp1_cell_max_area",
                        help="Cell maximum area for the first dispatching",
                        type=positive_float,
                        default=2000)
    parser.add_argument("--disp1_cell_min_circ",
                        help="Cell minimum circularity for the first dispatching",
                        type=positive_float,
                        default=.70)
    parser.add_argument("--disp1_clust_min_cell_nb",
                        help="Minimum number of cells in a cluster for the first dispatching",
                        type=positive_int,
                        default=3)
    parser.add_argument("--disp2_cell_min_area",
                        help="Cell minimum area for the second dispatching",
                        type=positive_float,
                        default=800)
    parser.add_argument("--disp2_cell_max_area",
                        help="Cell maximum area for the second dispatching",
                        type=positive_float,
                        default=4000)
    parser.add_argument("--disp2_cell_min_circ",
                        help="Cell minimum circularity for the second dispatching",
                        type=positive_float,
                        default=.85)
    parser.add_argument("--disp2_clust_min_cell_nb",
                        help="Minimum number of cells in a cluster for the second dispatching",
                        type=positive_float,
                        default=1)


    args = parser.parse_args(args=argv)


    with open(args.cell_classifier, "rb") as cell_classif_file:
        cell_classes = pickle.load(cell_classif_file)
        cell_model = pickle.load(cell_classif_file)
    with open(args.pattern_classifier, "rb") as pattern_classif_file:
        arch_pattern_classes = pickle.load(pattern_classif_file)
        arch_pattern_model = pickle.load(pattern_classif_file)

    args.cell_classifier = cell_model
    args.pattern_classifier = arch_pattern_model

    #----------Building the objects & Running----------------#

    with ThyroidJob(**vars(args)) as job:
        job.run()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

