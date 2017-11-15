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
from cell_counting.utils import make_dirs, check_default, params_remove_list

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
    parser.add_argument('--cytomine_force_download', dest='cytomine_force_download', type=bool, default=True,
                        help="Force download from Cytomine or not")

    # Objects
    parser.add_argument('--cytomine_object_term', dest='cytomine_object_term', type=int,
                        help="The Cytomine identifier of object term")
    parser.add_argument('--cytomine_object_user', dest='cytomine_object_user', type=int,
                        help="The Cytomine identifier of object owner")
    parser.add_argument('--cytomine_object_reviewed_only', dest='cytomine_object_reviewed_only', type=bool,
                        help="Whether objects have to be reviewed or not")

    # ROI
    parser.add_argument('--cytomine_roi_term', dest='cytomine_roi_term', type=int, default=None,
                        help="The Cytomine identifier of region of interest term")
    parser.add_argument('--cytomine_roi_user', dest='cytomine_roi_user', type=int,
                        help="The Cytomine identifier of ROI owner")
    parser.add_argument('--cytomine_roi_reviewed_only', dest='cytomine_roi_reviewed_only', type=bool,
                        help="Whether ROIs have to be reviewed or not")

    # Pre-processing
    parser.add_argument('--mean_radius', dest='mean_radius', type=int, required=True,
                        help="The mean radius of object to detect")
    parser.add_argument('--pre_transformer', dest='pre_transformer',
                        default=None, choices=['edt', 'euclidean_distance_transform', 'density', None, 'None'],
                        help="Scoremap transformer (None, edt, euclidean_distance_transform, density)")
    parser.add_argument('--pre_alpha', dest='pre_alpha', action='append', type=int,
                        help="Exponential decrease rate of distance (if EDT)")

    # Subwindows for training
    parser.add_argument('--sw_input_size', dest='sw_input_size', action='append', type=int,
                        help="Size of input subwindow")
    parser.add_argument('--sw_colorspace', dest="sw_colorspace", type=str, default='RGB__rgb',
                        help="List of colorspace features")
    parser.add_argument('--sw_extr_npi', dest="sw_extr_npi", action='append', type=int,
                        help="Number of extracted subwindows per image "
                             "(if 'random' mode)")

    # CNN
    parser.add_argument('--cnn_architecture', '--architecture', dest='cnn_architecture',
                        type=str, choices=['FCRN-A', 'FCRN-B', 'FCRN-test'])
    parser.add_argument('--cnn_initializer', '--initializer', dest='cnn_initializer',
                        action='append', type=str)
    parser.add_argument('--cnn_regularizer', '--regularizer', dest='cnn_regularizer',
                        action='append', type=str)
    parser.add_argument('--cnn_batch_normalization', '--batch_normalization', dest='cnn_batch_normalization',
                        action='append', type=bool)
    parser.add_argument('--cnn_learning_rate', '--learning_rate', '--lr', dest='cnn_learning_rate',
                        action='append', type=float)
    parser.add_argument('--cnn_momentum', '--momentum', dest='cnn_momentum',
                        action='append', type=float)
    parser.add_argument('--cnn_nesterov', '--nesterov', dest='cnn_nesterov',
                        action='append', type=bool)
    parser.add_argument('--cnn_decay', '--decay', dest='cnn_decay',
                        action='append', type=float)
    parser.add_argument('--cnn_epochs', '--epochs', dest='cnn_epochs',
                        action='append', type=int)
    parser.add_argument('--cnn_batch_size', '--batch_size', dest='cnn_batch_size',
                        action='append', type=int)

    # Dataset augmentation
    parser.add_argument('--augmentation', dest='augmentation', type=bool)
    parser.add_argument('--aug_rotation_range', dest='rotation_range', type=float)
    parser.add_argument('--aug_width_shift_range', dest='width_shift_range', type=float)
    parser.add_argument('--aug_height_shift_range', dest='height_shift_range', type=float)
    parser.add_argument('--aug_zoom_range', dest='zoom_range', type=float)
    parser.add_argument('--aug_fill_mode', dest='fill_mode', type=str)
    parser.add_argument('--aug_horizontal_flip', dest='horizontal_flip', type=bool)
    parser.add_argument('--aug_vertical_flip', dest='vertical_flip', type=bool)
    parser.add_argument('--aug_featurewise_center', dest='featurewise_center', type=bool)
    parser.add_argument('--aug_featurewise_std_normalization', dest='featurewise_std_normalization', type=bool)

    # Execution
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help="Number of jobs")
    parser.add_argument('--verbose', '-v', dest='verbose', default=0, action='count', help="Level of verbosity")

    params, other = parser.parse_known_args(argv)
    if params.cytomine_working_path is None:
        params.cytomine_working_path = os.path.join(tempfile.gettempdir(), "cytomine")
    make_dirs(params.cytomine_working_path)

    params.pre_transformer = check_default(params.pre_transformer, None, return_list=False)
    params.pre_alpha = check_default(params.pre_alpha, 5)
    params.cnn_architecture = check_default(params.cnn_architecture, 'FCRN-A', return_list=False)
    params.cnn_initializer = check_default(params.cnn_initializer, 'orthogonal')
    params.cnn_regularizer = check_default(params.cnn_regularizer, None)
    params.cnn_batch_normalization = check_default(params.cnn_batch_normalization, True)
    params.cnn_learning_rate = check_default(params.cnn_learning_rate, 0.02)
    params.cnn_momentum = check_default(params.cnn_momentum, 0.9)
    params.cnn_nesterov = check_default(params.cnn_nesterov, True)
    params.cnn_decay = check_default(params.cnn_decay, 0.)
    params.cnn_epochs = check_default(params.cnn_epochs, 3)
    params.cnn_batch_size = check_default(params.cnn_batch_size, 2)

    d = 8 if params.cnn_architecture == 'FCRN-A' else 4
    params.sw_input_size = check_default(params.sw_input_size, 4)
    params.sw_input_size = [((s // d * d + d), (s // d * d + d)) for s in params.sw_input_size]
    params.sw_output_size = params.sw_input_size
    params.sw_size = params.sw_input_size
    params.sw_colorspace = params.sw_colorspace.split(' ')
    params.sw_extr_npi = check_default(params.sw_extr_npi, 100)
    params.sw_extr_mode = ['random']

    params.augmentation = check_default(params.augmentation, True, return_list=False)
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
    else:
        params.rotation_range = 0.
        params.width_shift_range = 0.
        params.height_shift_range = 0.
        params.zoom_range = 0.
        params.fill_mode = 'reflect'
        params.horizontal_flip = False
        params.vertical_flip = False
        params.featurewise_center = False
        params.featurewise_std_normalization = False

    params = params_remove_list(params)

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
        print (model_path)

        # Callbacks
        checkpoint_callback = ModelCheckpoint(model_file, monitor='loss', save_best_only=True)
        lr_callback = LearningRateScheduler(lr_scheduler)
        callbacks = [checkpoint_callback, lr_callback]

        cytomine.update_job_status(job.job, status_comment="Training FCRN...", progress=5)
        estimator = FCRN(FCRN.build_fcrn, callbacks, **vars(params))
        estimator.fit(np.asarray(X), np.asarray(y))

        cytomine.update_job_status(job.job, status_comment="Saving (best) model", progress=95)
        estimator.save(model_file)

        cytomine.update_job_status(job.job, status_comment="Finished.", progress=100)


if __name__ == '__main__':
    import sys
    train(sys.argv[1:])