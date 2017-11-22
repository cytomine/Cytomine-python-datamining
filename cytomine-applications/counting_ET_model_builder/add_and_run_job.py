# -*- coding: utf-8 -*-
import os
import tempfile
from argparse import ArgumentParser

import numpy as np
from cytomine import Cytomine
from cytomine_utilities import CytomineJob
from sldc import StandardOutputLogger, Logger

from cell_counting.cytomine_utils import get_dataset
from cell_counting.utils import make_dirs, check_default, params_remove_list, check_max_features, params_remove_none, \
    str2bool, str2int, str2float
from cell_counting.extratrees_methods import CellCountRandomizedTrees

__author__ = "Ulysse Rubens <urubens@uliege.be>"
__version__ = "0.1"


def train(argv):
    parser = ArgumentParser(prog="Extra-Trees Object Counter Model Builder")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='cytomine_host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='cytomine_public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='cytomine_private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_base_path', dest='cytomine_base_path',
                        default='/api/', help="The Cytomine base path")
    parser.add_argument('--cytomine_working_path', dest='cytomine_working_path',
                        default=None, help="The working directory (eg: /tmp)")
    parser.add_argument('--cytomine_id_software', dest='cytomine_software', type=int,
                        help="The Cytomine software identifier")
    parser.add_argument('--cytomine_id_project', dest='cytomine_project', type=int,
                        help="The Cytomine project identifier")
    parser.add_argument('--cytomine_force_download', dest='cytomine_force_download', type=str, default=True,
                        help="Force download from Cytomine or not")

    # Objects
    parser.add_argument('--cytomine_object_term', dest='cytomine_object_term', type=int,
                        help="The Cytomine identifier of object term")
    parser.add_argument('--cytomine_object_user', dest='cytomine_object_user', type=str,
                        help="The Cytomine identifier of object owner")
    parser.add_argument('--cytomine_object_reviewed_only', dest='cytomine_object_reviewed_only', type=str,
                        help="Whether objects have to be reviewed or not")

    # ROI
    parser.add_argument('--cytomine_roi_term', dest='cytomine_roi_term', type=int, default=None,
                        help="The Cytomine identifier of region of interest term")
    parser.add_argument('--cytomine_roi_user', dest='cytomine_roi_user', type=str,
                        help="The Cytomine identifier of ROI owner")
    parser.add_argument('--cytomine_roi_reviewed_only', dest='cytomine_roi_reviewed_only', type=str,
                        help="Whether ROIs have to be reviewed or not")

    # Pre-processing
    parser.add_argument('--mean_radius', dest='mean_radius', type=int, required=True,
                        help="The mean radius of object to detect")
    parser.add_argument('--pre_transformer', dest='pre_transformer',
                        default=None, choices=['edt', 'euclidean_distance_transform', 'density', '', None, 'None'],
                        help="Scoremap transformer (None, edt, euclidean_distance_transform, density)")
    parser.add_argument('--pre_alpha', dest='pre_alpha', default=3,
                        help="Exponential decrease rate of distance (if EDT)")

    # Subwindows
    parser.add_argument('--sw_input_size', dest='sw_input_size', type=int, default=8,
                        help="Size of input subwindow")
    parser.add_argument('--sw_output_size', dest='sw_output_size', type=int, default=1,
                        help="Size of output subwindow (ignored for FCRN)")
    parser.add_argument('--sw_extr_mode', dest='sw_extr_mode', choices=['random', 'sliding', 'scoremap_constrained'],
                        default='random', help="Mode of extraction (random, scoremap_constrained)")
    parser.add_argument('--sw_extr_score_thres', dest='sw_extr_score_thres', default=0.4,
                        help="Minimum threshold to be foreground in subwindows extraction"
                             "(if 'scoremap_constrained' mode)")
    parser.add_argument('--sw_extr_ratio', dest='sw_extr_ratio', default=0.5,
                        help="Ratio of background subwindows extracted in subwindows "
                             "extraction (if 'scoremap_constrained' mode)")
    parser.add_argument('--sw_extr_npi', dest="sw_extr_npi", default=100,
                        help="Number of extracted subwindows per image (if 'random' mode)")
    parser.add_argument('--sw_colorspace', dest="sw_colorspace", type=str, default='RGB__rgb',
                        help="List of colorspace features")

    # Forest
    parser.add_argument('--forest_method', dest='forest_method', type=str,
                        default='ET-regr', choices=['ET-clf', 'ET-regr', 'RF-clf', 'RF-regr'],
                        help="Type of forest method")
    parser.add_argument('--forest_n_estimators', dest='forest_n_estimators', default=10, type=int,
                        help="Number of trees in forest")
    parser.add_argument('--forest_min_samples_split', dest='forest_min_samples_split', default=10, type=int,
                        help="Minimum number of samples for further splitting")
    parser.add_argument('--forest_max_features', dest='forest_max_features', default='sqrt',
                        help="Max features")

    # Dataset augmentation
    parser.add_argument('--augmentation', dest='augmentation', type=str, default=False)
    parser.add_argument('--aug_rotation_range', dest='rotation_range', type=float, default=0.)
    parser.add_argument('--aug_width_shift_range', dest='width_shift_range', type=float, default=0.)
    parser.add_argument('--aug_height_shift_range', dest='height_shift_range', type=float, default=0.)
    parser.add_argument('--aug_zoom_range', dest='zoom_range', type=float, default=0.)
    parser.add_argument('--aug_fill_mode', dest='fill_mode', type=str, default="reflect")
    parser.add_argument('--aug_horizontal_flip', dest='horizontal_flip', type=bool, default=False)
    parser.add_argument('--aug_vertical_flip', dest='vertical_flip', type=bool, default=False)
    parser.add_argument('--aug_featurewise_center', dest='featurewise_center', type=bool, default=False)
    parser.add_argument('--aug_featurewise_std_normalization', dest='featurewise_std_normalization', type=bool,
                        default=False)

    # Execution
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help="Number of jobs")
    parser.add_argument('--verbose', '-v', dest='verbose', default=0, type=int, help="Level of verbosity")

    params, other = parser.parse_known_args(argv)
    if params.cytomine_working_path is None:
        params.cytomine_working_path = os.path.join(tempfile.gettempdir(), "cytomine")
    make_dirs(params.cytomine_working_path)

    params.cytomine_force_download = str2bool(params.cytomine_force_download)
    params.cytomine_object_reviewed_only = str2bool(params.cytomine_object_reviewed_only)
    params.cytomine_roi_reviewed_only = str2bool(params.cytomine_roi_reviewed_only)
    params.cytomine_object_user = str2int(params.cytomine_object_user)
    params.cytomine_roi_user = str2int(params.cytomine_roi_user)
    params.pre_alpha = str2int(params.pre_alpha)
    params.sw_extr_score_thres = str2float(params.sw_extr_score_thres)
    params.sw_extr_ratio = str2float(params.sw_extr_ratio)
    params.sw_extr_npi = str2int(params.sw_extr_npi)
    params.forest_max_features = check_max_features(params.forest_max_features)[0]
    params.sw_input_size = (params.sw_input_size, params.sw_input_size)
    params.sw_output_size = (params.sw_output_size, params.sw_output_size)
    params.sw_colorspace = params.sw_colorspace.split('+')

    # Initialize logger
    logger = StandardOutputLogger(params.verbose)
    for key, val in sorted(vars(params).iteritems()):
        logger.info("[PARAMETER] {}: {}".format(key, val))

    # Initialize Cytomine client
    cytomine = Cytomine(
        params.cytomine_host,
        params.cytomine_public_key,
        params.cytomine_private_key,
        working_path=params.cytomine_working_path,
        base_path=params.cytomine_base_path,
        verbose=(params.verbose >= Logger.DEBUG)
    )

    # Start job
    with CytomineJob(cytomine,
                     params.cytomine_software,
                     params.cytomine_project,
                     parameters=vars(params_remove_none(params))) as job:
        cytomine.update_job_status(job.job, status_comment="Starting...", progress=0)

        cytomine.update_job_status(job.job, status_comment="Loading training set...", progress=1)
        X, y = get_dataset(cytomine, params.cytomine_working_path, params.cytomine_project, params.cytomine_object_term,
                           params.cytomine_roi_term, params.cytomine_object_user, params.cytomine_object_reviewed_only,
                           params.cytomine_roi_user, params.cytomine_roi_reviewed_only, params.cytomine_force_download)
        logger.d("X size: {} samples".format(len(X)))
        logger.d("y size: {} samples".format(len(y)))

        cytomine.update_job_status(job.job, status_comment="Training forest...", progress=5)
        estimator = CellCountRandomizedTrees(logger=logger, **vars(params))
        estimator.fit(np.asarray(X), np.asarray(y))

        cytomine.update_job_status(job.job, status_comment="Saving (best) model", progress=95)
        model_path = os.path.join(params.cytomine_working_path, "models", str(params.cytomine_software))
        model_file = os.path.join(model_path, "{}.pkl".format(job.job.id))
        make_dirs(model_path)
        estimator.save(model_file)

        cytomine.update_job_status(job.job, status_comment="Finished.", progress=100)


if __name__ == '__main__':
    import sys

    train(sys.argv[1:])
