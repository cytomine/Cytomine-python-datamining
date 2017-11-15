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


def install_CNN_ObjectCounter_Predictor(cytomine, software_router, software_path, software_working_path):
    if software_path is not None:
        software_path = os.path.join(software_path, "object_counter_predictor/CNN/add_and_run_job.py")

    if software_working_path is not None:
        software_working_path = os.path.join(software_working_path, "object_counter")

    software = InstallSoftware("ObjectCounter_CNN_Predictor", "pyxitSuggestedTermJobService", "Default",
                               software_router, software_path, software_working_path)
    software.add_parameter("cytomine_id_software", int, 0, required=True, set_by_server=True)
    software.add_parameter("cytomine_id_project", int, 0, required=True, set_by_server=True)

    software.add_parameter("model_id_job", "Domain", "", required=True, set_by_server=False,
                           uri="/api/job.json?project=$currentProject$", uri_print_attr="softwareName",
                           uri_sort_attr="softwareName")

    software.add_parameter("cytomine_object_term", "Domain", "", required=True,
                           uri="/api/project/$currentProject$/term.json", uri_print_attr="name", uri_sort_attr="name")

    software.add_parameter("image", "ListDomain", "", required=False, set_by_server=False,
                           uri="/api/project/$currentProject$/imageinstance.json", uri_print_attr="instanceFilename",
                           uri_sort_attr="instanceFilename")
    software.add_parameter("annotation", "Number", "", required=False, set_by_server=False)

    software.add_parameter("post_threshold", float, 0.5, required=True, set_by_server=False)
    software.add_parameter("post_sigma", float, 0.0, required=True, set_by_server=False)
    software.add_parameter("post_min_dist", int, 5, required=True, set_by_server=False)

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

    install_CNN_ObjectCounter_Predictor(conn, params.software_router, params.software_path,
                                        params.software_working_path)