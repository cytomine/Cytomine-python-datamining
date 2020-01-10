# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
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
# */


__author__ = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__ = ["Stévens Benjamin <b.stevens@ulg.ac.be>", "Elodie Burtin <elodie.burtin@cytomine.coop"]
__copyright__ = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"
__version__ = "2.0.0"

import logging

from cytomine import Cytomine
from cytomine.models import Software, SoftwareParameter, SoftwareProject

# Connection parameters
cytomine_host = "XXX"
cytomine_public_key = "XXX"
cytomine_private_key = "XXX"
id_project = 0

# Connection to Cytomine Core
Cytomine.connect(cytomine_host, cytomine_public_key, cytomine_private_key, verbose=logging.INFO)

execute_command = ("python algo/segmentation_model_builder/add_and_run_job.py " +
                   "--cytomine_host $host " +
                   "--cytomine_public_key $publicKey " +
                   "--cytomine_private_key $privateKey " +
                   "--cytomine_id_software $cytomine_id_software " +
                   "--cytomine_id_project $cytomine_id_project " +
                   "--cytomine_annotation_projects $cytomine_annotation_projects " +
                   "--cytomine_zoom_level $cytomine_zoom_level " +
                   "--cytomine_predict_terms $cytomine_predict_terms " +
                   "--cytomine_excluded_terms $cytomine_excluded_terms " +
                   "--pyxit_target_width $pyxit_target_width " +
                   "--pyxit_target_height $pyxit_target_height " +
                   "--pyxit_colorspace $pyxit_colorspace " +
                   "--pyxit_n_jobs $pyxit_n_jobs " +
                   "--pyxit_save_to $pyxit_save_to " +
                   "--pyxit_transpose $pyxit_transpose " +
                   "--pyxit_fixed_size $pyxit_fixed_size " +
                   "--pyxit_interpolation $pyxit_interpolation " +
                   "--forest_n_estimators $forest_n_estimators " +
                   "--forest_max_features $forest_max_features " +
                   "--forest_min_samples_split $forest_min_samples_split " +
                   "--pyxit_n_subwindows $pyxit_n_subwindows " +
                   "--cytomine_reviewed $cytomine_reviewed " +
                   "--log_level INFO")

# define software parameter template
software = Software("Segmentation_Model_Builder", "createRabbitJobWithArgsService",
                    "ValidateAnnotation", execute_command).save()

SoftwareParameter("cytomine_id_software", "Number", software.id,  0, True, 400, True).save()
SoftwareParameter("cytomine_id_project", "Number", software.id, 0, True, 500, True).save()
SoftwareParameter("pyxit_save_to", "String", software.id, "", True, 400, True).save()
SoftwareParameter("pyxit_target_width", "Number", software.id, 24, True, 500, False).save()
SoftwareParameter("pyxit_target_height", "Number", software.id, 24, True, 600, False).save()
SoftwareParameter("pyxit_n_subwindows", "Number", software.id, 100, True, 200, False).save()
SoftwareParameter("pyxit_colorspace", "Number", software.id, 2, True, 900, False).save()
SoftwareParameter("pyxit_interpolation", "Number", software.id, 1, True, 700, False).save()
SoftwareParameter("pyxit_transpose", "Boolean", software.id, "false", True, 800, False).save()
SoftwareParameter("pyxit_fixed_size", "Boolean", software.id, "false", True, 905, False).save()
SoftwareParameter("forest_n_estimators", "Number", software.id, 10, True, 1100, False).save()
SoftwareParameter("forest_max_features", "Number", software.id, 28, True, 1200, False).save()
SoftwareParameter("forest_min_samples_split", "Number", software.id, 2, True, 1300, False).save()
SoftwareParameter("pyxit_n_jobs", "Number", software.id, 10, True, 1000, False).save()
SoftwareParameter("cytomine_annotation_projects", "ListDomain", software.id, "", True, 1200, False,
                  "/api/ontology/$currentOntology$/project.json", "name", "name").save()
SoftwareParameter("cytomine_predict_terms", "ListDomain", software.id, "", True, 1300, False,
                  "/api/project/$currentProject$/term.json", "name", "name").save()
SoftwareParameter("cytomine_excluded_terms", "ListDomain", software.id, "", False, 1400, False,
                  "/api/project/$currentProject$/term.json", "name", "name").save()
SoftwareParameter("cytomine_zoom_level", "Number", software.id, 0, True, 1500, False).save()
SoftwareParameter("cytomine_reviewed", "Boolean", software.id, "false", True, 1600, False).save()

# add software to a given project
if id_project:
    SoftwareProject(software.id, id_project).save()
