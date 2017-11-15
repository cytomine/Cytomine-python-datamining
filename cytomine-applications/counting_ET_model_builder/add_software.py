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


def install_ET_ObjectCounter_Model_builder(cytomine, software_router, software_path, software_working_path):
    if software_path is not None:
        software_path = os.path.join(software_path, "object_counter_model_builder/ET/add_and_run_job.py")

    if software_working_path is not None:
        software_working_path = os.path.join(software_working_path, "object_counter")

    software = InstallSoftware("ObjectCounter_ET_Model_Builder", "pyxitSuggestedTermJobService", "Default",
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

    software.add_parameter("sw_input_size", int, default_value=8, required=True)
    software.add_parameter("sw_output_size", int, default_value=1, required=True)
    software.add_parameter("sw_extr_mode", str, default_value="random", required=True)
    software.add_parameter("sw_extr_score_thres", float, default_value="", required=False)
    software.add_parameter("sw_extr_ratio", float, default_value="", required=False)
    software.add_parameter("sw_extr_npi", float, default_value="", required=False)
    software.add_parameter("sw_colorspace", str, required=True,
                           default_value="RGB__rgb RGB__Luv RGB__hsv L__normalized L__sobel1 L__gradmagn")
    software.add_parameter("forest_method", str, default_value="ET-regr", required=True)
    software.add_parameter("forest_n_estimators", int, default_value=10, required=True)
    software.add_parameter("forest_min_samples_split", int, default_value=2, required=True)
    software.add_parameter("forest_max_features", str, default_value="sqrt", required=True)
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

    install_ET_ObjectCounter_Model_builder(conn, params.software_router, params.software_path,
                                           params.software_working_path)