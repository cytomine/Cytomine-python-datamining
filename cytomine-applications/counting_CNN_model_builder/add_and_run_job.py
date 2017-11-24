# -*- coding: utf-8 -*-

import os
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
import tempfile
from argparse import ArgumentParser

import numpy as np
from cytomine import Cytomine
from cytomine_utilities import CytomineJob
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sldc import StandardOutputLogger, Logger

from cell_counting.cnn_methods import lr_scheduler, FCRN
from cell_counting.cytomine_utils import get_dataset
from cell_counting.utils import make_dirs, check_default, str2bool

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"


def train(argv):
    parser = ArgumentParser(prog="CNN Object Counter Model Builder")

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
    parser.add_argument('--pre_transformer', dest='pre_transformer',
                        default='density', choices=['edt', 'euclidean_distance_transform', 'density', None, 'None'],
                        help="Scoremap transformer (None, edt, euclidean_distance_transform, density)")
    parser.add_argument('--pre_alpha', dest='pre_alpha', type=int, default=3,
                        help="Exponential decrease rate of distance (if EDT)")

    # Subwindows for training
    parser.add_argument('--sw_input_size', dest='sw_input_size', type=int, default=128,
                        help="Size of input subwindow")
    parser.add_argument('--sw_colorspace', dest="sw_colorspace", type=str, default='RGB__rgb',
                        help="List of colorspace features")
    parser.add_argument('--sw_extr_npi', dest="sw_extr_npi", type=int, default=100,
                        help="Number of extracted subwindows per image (if 'random' mode)")

    # CNN
    parser.add_argument('--cnn_architecture', '--architecture', dest='cnn_architecture',
                        type=str, choices=['FCRN-A', 'FCRN-B'], default='FCRN-A')
    parser.add_argument('--cnn_initializer', '--initializer', dest='cnn_initializer', type=str, default='orthogonal')
    parser.add_argument('--cnn_batch_normalization', '--batch_normalization', dest='cnn_batch_normalization', type=str, default=True)
    parser.add_argument('--cnn_learning_rate', '--learning_rate', '--lr', dest='cnn_learning_rate', type=float, default=0.01)
    parser.add_argument('--cnn_momentum', '--momentum', dest='cnn_momentum', type=float, default=0.9)
    parser.add_argument('--cnn_nesterov', '--nesterov', dest='cnn_nesterov', type=str, default=True)
    parser.add_argument('--cnn_decay', '--decay', dest='cnn_decay', type=float, default=0.0)
    parser.add_argument('--cnn_epochs', '--epochs', dest='cnn_epochs', type=int, default=24)
    parser.add_argument('--cnn_batch_size', '--batch_size', dest='cnn_batch_size', type=int, default=16)

    # Dataset augmentation
    parser.add_argument('--augmentation', dest='augmentation', type=str, default=True)
    parser.add_argument('--aug_rotation_range', dest='rotation_range', type=float, default=0.)
    parser.add_argument('--aug_width_shift_range', dest='width_shift_range', type=float, default=0.)
    parser.add_argument('--aug_height_shift_range', dest='height_shift_range', type=float, default=0.)
    parser.add_argument('--aug_zoom_range', dest='zoom_range', type=float, default=0.)
    parser.add_argument('--aug_fill_mode', dest='fill_mode', type=str, default="reflect")
    parser.add_argument('--aug_horizontal_flip', dest='horizontal_flip', type=bool, default=False)
    parser.add_argument('--aug_vertical_flip', dest='vertical_flip', type=bool, default=False)
    parser.add_argument('--aug_featurewise_center', dest='featurewise_center', type=bool, default=False)
    parser.add_argument('--aug_featurewise_std_normalization', dest='featurewise_std_normalization', type=bool, default=False)

    # Execution
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help="Number of jobs")
    parser.add_argument('--verbose', '-v', dest='verbose', type=int, default=0, help="Level of verbosity")

    params, other = parser.parse_known_args(argv)
    if params.cytomine_working_path is None:
        params.cytomine_working_path = os.path.join(tempfile.gettempdir(), "cytomine")
    make_dirs(params.cytomine_working_path)

    params.cytomine_force_download = str2bool(params.cytomine_force_download)
    params.cytomine_object_reviewed_only = str2bool(params.cytomine_object_reviewed_only)
    params.cytomine_roi_reviewed_only = str2bool(params.cytomine_roi_reviewed_only)
    params.cnn_batch_normalization = str2bool(params.cnn_batch_normalization)
    params.cnn_nesterov = str2bool(params.cnn_nesterov)
    params.augmentation = str2bool(params.augmentation)

    d = 8. if params.cnn_architecture == 'FCRN-A' else 4.
    params.sw_size = (int(np.ceil(params.sw_input_size/d)+d), int(np.ceil(params.sw_input_size/d)+d))
    params.sw_input_size = params.sw_size
    params.sw_output_size = params.sw_size
    params.sw_colorspace = params.sw_colorspace.split(' ')
    params.sw_extr_mode = 'random'
    params.cnn_regularizer = None
    params.mean_radius = 2

    if params.augmentation:
        params.rotation_range = check_default(params.rotation_range, 30., return_list=False)
        params.width_shift_range = check_default(params.width_shift_range, 0.3, return_list=False)
        params.height_shift_range = check_default(params.height_shift_range, 0.3, return_list=False)
        params.zoom_range = check_default(params.zoom_range, 0.3, return_list=False)
        params.fill_mode = check_default(params.fill_mode, 'constant', return_list=False)
        params.horizontal_flip = check_default(params.horizontal_flip, True, return_list=False)
        params.vertical_flip = check_default(params.vertical_flip, True, return_list=False)
        params.featurewise_center = check_default(params.featurewise_center, False, return_list=False)
        params.featurewise_std_normalization = check_default(params.featurewise_std_normalization, False,
                                                             return_list=False)

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
                     parameters=vars(params)) as job:
        cytomine.update_job_status(job.job, status_comment="Starting...", progress=0)

        cytomine.update_job_status(job.job, status_comment="Loading training set...", progress=1)
        logger.i("Loading training set...")
        X, y = get_dataset(cytomine, params.cytomine_working_path, params.cytomine_project, params.cytomine_object_term,
                           params.cytomine_roi_term, params.cytomine_object_user, params.cytomine_object_reviewed_only,
                           params.cytomine_roi_user, params.cytomine_roi_reviewed_only, params.cytomine_force_download)
        logger.d("X size: {} samples".format(len(X)))
        logger.d("y size: {} samples".format(len(y)))

        # Rename parameters
        params.architecture = params.cnn_architecture
        params.initializer = params.cnn_initializer
        params.regularizer = params.cnn_regularizer
        params.batch_normalization = params.cnn_batch_normalization
        params.learning_rate = params.cnn_learning_rate
        params.momentum = params.cnn_momentum
        params.nesterov = params.cnn_nesterov
        params.decay = params.cnn_decay
        params.epochs = params.cnn_epochs
        params.batch_size = params.cnn_batch_size

        model_path = os.path.join(params.cytomine_working_path, "models", str(params.cytomine_software))
        model_file = os.path.join(model_path, "{}.h5".format(job.job.id))
        make_dirs(model_path)

        # Callbacks
        # checkpoint_callback = ModelCheckpoint(model_file, monitor='loss', save_best_only=True)
        lr_callback = LearningRateScheduler(lr_scheduler)
        callbacks = [lr_callback]

        logger.i("Training FCRN...")
        cytomine.update_job_status(job.job, status_comment="Training FCRN...", progress=5)
        estimator = FCRN(FCRN.build_fcrn, callbacks, **vars(params))
        estimator.fit(np.asarray(X), np.asarray(y))

        logger.i("Saving model...")
        cytomine.update_job_status(job.job, status_comment="Saving (best) model", progress=95)
        estimator.save(model_file)

        logger.i("Finished.")
        cytomine.update_job_status(job.job, status_comment="Finished.", progress=100)


if __name__ == '__main__':
    import sys
    train(sys.argv[1:])