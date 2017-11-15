# -*- coding: utf-8 -*-

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

import os
import pickle
import tempfile
from argparse import ArgumentParser

import numpy as np
from cytomine import Cytomine
from cytomine.models import AnnotationCollection, Annotation
from cytomine_utilities import CytomineJob
from sldc import StandardOutputLogger, Logger

from cell_counting.utils import make_dirs, params_remove_none
from cell_counting.cytomine_utils import upload_annotations

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"


def predict(argv):
    parser = ArgumentParser(prog="Extra-Trees Object Counter Predictor")

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

    # Objects
    parser.add_argument('--cytomine_object_term', dest='cytomine_object_term', type=int,
                        help="The Cytomine identifier of object term")

    # Post-processing
    parser.add_argument('--post_threshold', dest='post_threshold', type=float,
                        help="Post-processing discarding threshold")
    parser.add_argument('--post_sigma', dest='post_sigma', type=float,
                        help="Std-dev of Gauss filter applied to smooth prediction")
    parser.add_argument('--post_min_dist', dest='post_min_dist', type=int,
                        help="Minimum distance between two peaks")

    # ROI
    parser.add_argument('--annotation', dest='annotation', type=int, default=None)
    parser.add_argument('--image', dest='image', type=int, action='append', default=None)

    # Execution
    parser.add_argument('--n_jobs', dest='n_jobs', type=int, default=1, help="Number of jobs")
    parser.add_argument('--verbose', '-v', dest='verbose', default=0, help="Level of verbosity")
    parser.add_argument('--model_id_job', dest='model_id_job', type=int, default=None, help="Model job ID")

    params, other = parser.parse_known_args(argv)
    if params.cytomine_working_path is None:
        params.cytomine_working_path = os.path.join(tempfile.gettempdir(), "cytomine")
    make_dirs(params.cytomine_working_path)

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

        cytomine.update_job_status(job.job, status_comment="Loading model...", progress=1)
        model_job = cytomine.get_job(params.model_id_job)
        model_file = os.path.join(params.cytomine_working_path, "models", str(model_job.software),
                                  "{}.pkl".format(model_job.id))
        with open(model_file, 'rb') as f:
            estimator = pickle.load(f)

        cytomine.update_job_status(job.job, status_comment="Dumping annotations/images to predict...", progress=3)
        if params.annotation is not None:
            if not isinstance(params.annotation, list):
                params.annotation = list(params.annotation)
            annots = [cytomine.get_annotation(id) for id in params.annotation]
            annots_collection = AnnotationCollection()
            annots_collection._data = annots
            crops = cytomine.dump_annotations(annotations=annots_collection,
                                              dest_path=os.path.join(params.cytomine_working_path, "crops",
                                                                     str(params.cytomine_project)),
                                              desired_zoom=0,
                                              get_image_url_func=Annotation.get_annotation_alpha_crop_url)
            X = crops.data()
        elif params.image is not None:
            if not isinstance(params.image, list):
                params.image = list(params.image)

            image_instances = [cytomine.get_image_instance(id) for id in params.image]
            image_instances = cytomine.dump_project_images(id_project=params.cytomine_project,
                                                           dest_path="/imageinstances/",
                                                           image_instances=image_instances)
            X = image_instances
        else:
            X = []

        logger.d("X size: {} samples".format(len(X)))

        for i, x in enumerate(X):
            cytomine.update_job_status(job.job, status_comment="Predicting ID {}...".format(x.id),
                                       progress=5 + np.ceil(i / len(X)) * 95)
            y = estimator.predict([x.filename])

            cytomine.update_job_status(job.job, status_comment="Uploading annotations...")
            upload_annotations(cytomine, x, y, term=params.cytomine_object_term)

        cytomine.update_job_status(job.job, status_comment="Finished.", progress=100)


if __name__ == '__main__':
    import sys
    predict(sys.argv[1:])
