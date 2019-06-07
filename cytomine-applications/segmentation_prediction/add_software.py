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

# connect to cytomine

cytomine_host = "XXX"
cytomine_public_key = "XXX"
cytomine_private_key = "XXX"
id_project = 0

# Connection to Cytomine Core
Cytomine.connect(cytomine_host, cytomine_public_key, cytomine_private_key, verbose=logging.INFO)

execute_command = ("python algo/segmentation_prediction/image_prediction_wholeslide.py " +
                   "--cytomine_host $host " +
                   "--cytomine_public_key $publicKey " +
                   "--cytomine_private_key $privateKey " +
                   "--cytomine_id_software $cytomine_id_software " +
                   "--cytomine_id_project $cytomine_id_project " +
                    "--pyxit_load_from $pyxit_load_from " +

                   "--model_id_job $model_id_job "
                   "--cytomine_zoom_level $cytomine_zoom_level " +
                   "--cytomine_tile_size $cytomine_tile_size " +
                   "--cytomine_tile_min_stddev $cytomine_tile_min_stddev " +
                   "--cytomine_tile_max_mean $cytomine_tile_max_mean " +
                   "--cytomine_startx $cytomine_startx " +
                   "--cytomine_starty $cytomine_starty " +
                   "--cytomine_endx $cytomine_endx " +
                   "--cytomine_endy $cytomine_endy " +
                   "--cytomine_nb_jobs $cytomine_nb_jobs " +
                   "--cytomine_predict_term $cytomine_predict_term " +
                   "--cytomine_roi_term $cytomine_roi_term " +
                   "--cytomine_reviewed_roi $cytomine_reviewed_roi " +
                   "--pyxit_target_width $pyxit_target_width " +
                   "--pyxit_target_height $pyxit_target_height " +
                   "--pyxit_colorspace $pyxit_colorspace " +
                   "--pyxit_nb_jobs $pyxit_nb_jobs " +
                   "--cytomine_predict_step $cytomine_predict_step " +

                   "--cytomine_union $cytomine_union " +
                   "--cytomine_postproc $cytomine_postproc " +
                   "--cytomine_min_size $cytomine_min_size " +
                   "--cytomine_max_size $cytomine_max_size " +
                   "--cytomine_union_min_length $cytomine_union_min_length " +
                   "--cytomine_union_bufferoverlap $cytomine_union_bufferoverlap " +
                   "--cytomine_union_area $cytomine_union_area " +
                   "--cytomine_union_min_point_for_simplify $cytomine_union_min_point_for_simplify " +
                   "--cytomine_union_min_point $cytomine_union_min_point " +
                   "--cytomine_union_max_point $cytomine_union_max_point " +
                   "--cytomine_union_nb_zones_width $cytomine_union_nb_zones_width " +
                   "--cytomine_union_nb_zones_height $cytomine_union_nb_zones_height " +

                   "--cytomine_count $cytomine_count " +

                   "--pyxit_post_classification $pyxit_post_classification " +
                   "--pyxit_post_classification_save_to $pyxit_post_classification_save_to "
                   
                   "--log_level INFO")

# define software parameter template
software = Software("Segmentation_Model_Builder", "pyxitSuggestedTermJobService",
                    "ValidateAnnotation", execute_command).save()

SoftwareParameter("cytomine_id_software", "Number", software.id,  0, True, 400, True).save()
SoftwareParameter("cytomine_id_project", "Number", software.id, 0, True, 500, True).save()
SoftwareParameter("pyxit_load_from", "String", software.id, "", True, 220, True).save()

SoftwareParameter("model_id_job", "Domain", software.id, "", True, 0, False,
                  "/api/job.json?project=$currentProject$", "softwareName", "softwareName").save()
SoftwareParameter("cytomine_zoom_level", "Number", software.id, 0, True, 10, False).save()
SoftwareParameter("pyxit_target_width", "Number", software.id, 24, True, 20, False).save()
SoftwareParameter("pyxit_target_height", "Number", software.id, 24, True, 30, False).save()
SoftwareParameter("pyxit_colorspace", "Number", software.id, 2, True, 40, False).save()
SoftwareParameter("pyxit_nb_jobs", "Number", software.id, 10, True, 45, False).save()
SoftwareParameter("cytomine_tile_size", "Number", software.id, 512, True, 50, False).save()
SoftwareParameter("cytomine_tile_min_stddev", "Number", software.id, 5, True, 60, False).save()
SoftwareParameter("cytomine_tile_max_mean", "Number", software.id, 250, True, 70, False).save()
SoftwareParameter("cytomine_startx", "Number", software.id, 0, True, 80, False).save()
SoftwareParameter("cytomine_starty", "Number", software.id, 0, True, 90, False).save()
SoftwareParameter("cytomine_endx", "Number", software.id, 0, True, 91, False).save()
SoftwareParameter("cytomine_endy", "Number", software.id, 0, True, 92, False).save()
SoftwareParameter("cytomine_roi_term", "Number", software.id, "", True, 100, False,
                  "/api/project/$currentProject$/term.json", "name", "name").save()
SoftwareParameter("cytomine_reviewed_roi", "Boolean", software.id, "false", False, 101, False).save()
SoftwareParameter("cytomine_predict_term", "Number", software.id, "", True, 105, False,
                  "/api/project/$currentProject$/term.json", "name", "name").save()
SoftwareParameter("cytomine_predict_step", "Number", software.id, 8, True, 110, False).save()
SoftwareParameter("cytomine_min_size", "Number", software.id, 1000, True, 120, False).save()
SoftwareParameter("cytomine_max_size", "Number", software.id, 10000000, True, 130, False).save()
SoftwareParameter("cytomine_union", "Boolean", software.id, "true", True, 135, False).save()
SoftwareParameter("cytomine_union_min_length", "Number", software.id, 10, True, 140, False).save()
SoftwareParameter("cytomine_union_bufferoverlap", "Number", software.id, 5, True, 150, False).save()
SoftwareParameter("cytomine_union_area", "Number", software.id, 5000, True, 160, False).save()
SoftwareParameter("cytomine_union_min_point_for_simplify", "Number", software.id, 1000, True, 170, False).save()
SoftwareParameter("cytomine_union_min_point", "Number", software.id, 500, True, 180, False).save()
SoftwareParameter("cytomine_union_max_point", "Number", software.id, 1000, True, 190, False).save()
SoftwareParameter("cytomine_union_nb_zones_width", "Number", software.id, 5, True, 200, False).save()
SoftwareParameter("cytomine_union_nb_zones_height", "Number", software.id, 5, True, 210, False).save()
SoftwareParameter("cytomine_postproc", "Boolean", software.id, "true", False, 115, False).save()
SoftwareParameter("pyxit_post_classification", "Boolean", software.id, "tmp", False, 230, False).save()
SoftwareParameter("pyxit_post_classification_save_to", "String", software.id, "tmp", False, 240, False).save()
SoftwareParameter("cytomine_count", "Boolean", software.id, "false", False, 260, False).save()
SoftwareParameter("cytomine_nb_jobs", "Number", software.id, 10, True, 300, False).save()

# add software to a given project
if id_project:
    SoftwareProject(software.id, id_project).save()
