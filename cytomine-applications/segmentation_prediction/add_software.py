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


__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__    = ["Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import cytomine
import sys

#connect to cytomine

cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=XXX

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#define software parameter template
software = conn.add_software("Segmentation_Model_Predict", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_zoom_level", software.id, "Number", 0, True, 10, False)
conn.add_software_parameter("pyxit_target_width", software.id, "Number", 24, True, 20, False)
conn.add_software_parameter("pyxit_target_height", software.id, "Number", 24, True, 30, False)
conn.add_software_parameter("pyxit_colorspace", software.id, "Number", 2, True, 40, False)
conn.add_software_parameter("cytomine_tile_size", software.id, "Number", 512, True, 50, False)
conn.add_software_parameter("cytomine_tile_min_stddev", software.id, "Number", 5, True, 60, False)
conn.add_software_parameter("cytomine_tile_max_mean", software.id, "Number", 250, True, 70, False)
conn.add_software_parameter("cytomine_startx", software.id, "Number", 0, True, 80, False)
conn.add_software_parameter("cytomine_starty", software.id, "Number", 0, True, 90, False)
conn.add_software_parameter("cytomine_roi_term", software.id, "Number", 0, True, 100, False)
conn.add_software_parameter("cytomine_predict_step", software.id, "Number", 4, True, 110, False)
conn.add_software_parameter("cytomine_min_size", software.id, "Number", 1000, True, 120, False)
conn.add_software_parameter("cytomine_max_size", software.id, "Number", 10000000, True, 130, False)
conn.add_software_parameter("cytomine_union_min_length", software.id, "Number", 10, True, 140, False)
conn.add_software_parameter("cytomine_union_bufferoverlap", software.id, "Number", 5, True, 150, False)
conn.add_software_parameter("cytomine_union_area", software.id, "Number", 5000, True, 160, False)
conn.add_software_parameter("cytomine_union_min_point_for_simplify", software.id, "Number", 1000, True, 170, False)
conn.add_software_parameter("cytomine_union_min_point", software.id, "Number", 500, True, 180, False)
conn.add_software_parameter("cytomine_union_max_point", software.id, "Number", 1000, True, 190, False)
conn.add_software_parameter("cytomine_union_nb_zones_width", software.id, "Number", 5, True, 200, False)
conn.add_software_parameter("cytomine_union_nb_zones_height", software.id, "Number", 5, True, 210, False)
conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 220, False)
conn.add_software_parameter("cytomine_postproc", software.id, "Boolean", "/tmp", False, 115, False)
conn.add_software_parameter("pyxit_post_classification", software.id, "Boolean", "/tmp", False, 230, False)
conn.add_software_parameter("pyxit_post_classification_save_to", software.id, "String", "/tmp", False, 240, False)
conn.add_software_parameter("cytomine_mask_internal_holes",software.id, "Boolean", "true", True, 250, False)
conn.add_software_parameter("cytomine_id_image", software.id, "Number", 0, True, 1, False)

#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)
