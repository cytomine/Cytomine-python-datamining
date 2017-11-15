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
import sys
from argparse import ArgumentParser

from cytomine import Cytomine

from cell_counting.cytomine_software import InstallSoftware

__author__ = "Rubens Ulysse <urubens@uliege.be>"
__copyright__ = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"


def install_CNN_ObjectCounter_Model_builder(cytomine, software_router, software_path, software_working_path):
    if software_path is not None:
        software_path = os.path.join(software_path, "object_counter_model_builder/CNN/add_and_run_job.py")

    if software_working_path is not None:
        software_working_path = os.path.join(software_working_path, "object_counter")

    software = InstallSoftware("ObjectCounter_CNN_Model_Builder", "pyxitSuggestedTermJobService", "Default",
                               software_router, software_path, software_working_path)
    software.add_parameter("cytomine_id_software", int, 0, required=True, set_by_server=True)
    software.add_parameter("cytomine_id_project", int, 0, required=True, set_by_server=True)

    software.add_parameter("cytomine_object_term", "Domain", "", required=True,
                           uri="/api/project/$currentProject$/term.json", uri_print_attr="name", uri_sort_attr="name")
    software.add_parameter("cytomine_object_user", "Domain", "", required=False,
                           uri="/api/project/$currentProject$/user.json", uri_print_attr="username", uri_sort_attr="username")
    software.add_parameter("cytomine_object_reviewed_only", bool, False, required=False)
    software.add_parameter("cytomine_roi_term", "Domain", "", required=True,
                           uri="/api/project/$currentProject$/term.json", uri_print_attr="name", uri_sort_attr="name")
    software.add_parameter("cytomine_roi_user", "Domain", "", required=False,
                           uri="/api/project/$currentProject$/user.json", uri_print_attr="username", uri_sort_attr="username")
    software.add_parameter("cytomine_roi_reviewed_only", bool, False, required=False)

    software.add_parameter("mean_radius", float, "", required=True)
    software.add_parameter("pre_transformer", str, default_value="", required=False)
    software.add_parameter("pre_alpha", int, default_value="", required=False)

    software.add_parameter("sw_input_size", int, default_value=128, required=True)
    software.add_parameter("sw_extr_npi", float, default_value=100, required=True)
    software.add_parameter("sw_colorspace", str, default_value="RGB__rgb", required=True)

    software.add_parameter("cnn_architecture", str, default_value="FCRN-A", required=True)
    software.add_parameter("cnn_initializer", str, default_value="orthogonal", required=True)
    software.add_parameter("cnn_regularizer", str, default_value="", required=True)
    software.add_parameter("cnn_batch_normalization", bool, default_value=False, required=True)
    software.add_parameter("cnn_learning_rate", float, default_value=0.02, required=True)
    software.add_parameter("cnn_momentum", float, default_value=0.9, required=True)
    software.add_parameter("cnn_decay", float, default_value=0.0, required=True)
    software.add_parameter("cnn_epochs", int, default_value=24, required=True)
    software.add_parameter("cnn_batch_size", int, default_value=32, required=True)
    software.add_parameter("augmentation", bool, default_value=True, required=True)

    software.add_parameter("n_jobs", int, default_value=1, required=True)
    software.add_parameter("verbose", int, default_value=3, required=False)

    cytomine_software = software.install_software(cytomine)
    print("New software ID is {}".format(cytomine_software.id))


if __name__ == "__main__":
    parser = ArgumentParser(prog="Software installer")
    parser.add_argument('--cytomine_host', type=str)
    parser.add_argument('--cytomine_public_key', type=str)
    parser.add_argument('--cytomine_private_key', type=str)
    parser.add_argument('--software_router', action="store_true")
    parser.add_argument('--software_path', type=str)
    parser.add_argument('--software_working_path', type=str)
    params, other = parser.parse_known_args(sys.argv[1:])

    # Connection to Cytomine Core
    conn = Cytomine(
        params.cytomine_host,
        params.cytomine_public_key,
        params.cytomine_private_key,
        base_path='/api/',
        working_path='/tmp',
        verbose=True
    )

    install_CNN_ObjectCounter_Model_builder(conn, params.software_router, params.software_path,
                                            params.software_working_path)