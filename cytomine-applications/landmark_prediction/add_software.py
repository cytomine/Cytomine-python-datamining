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


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine
import sys

#connect to cytomine : parameters to set
cytomine_host=""
cytomine_public_key=""
cytomine_private_key=""
id_project=0

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#define software parameter template
software = conn.add_software("Landmark_Model_Predictor", "pyxitSuggestedTermJobService","ValidateAnnotation")

conn.add_software_parameter("landmark_terms",           software.id, "List", None, True, 1 , False)
conn.add_software_parameter("landmark_r",               software.id, "List", None, True, 2 , False)
conn.add_software_parameter("landmark_rmax",            software.id, "List", None, True, 3 , False)
conn.add_software_parameter("landmark_p",               software.id, "List", None, True, 4 , False)
conn.add_software_parameter("landmark_npred",           software.id, "List", None, True, 5 , False)
conn.add_software_parameter("landmark_ntimes",          software.id, "List", None, True, 6 , False)
conn.add_software_parameter("landmark_alpha",           software.id, "List", None, True, 7 , False)
conn.add_software_parameter("landmark_depth",           software.id, "List", None, True, 8 , False)
conn.add_software_parameter("landmark_window_size",     software.id, "List", None, True, 8 , False)
conn.add_software_parameter("forest_n_estimators",      software.id, "List", None, True, 9 , False)
conn.add_software_parameter("forest_max_features",      software.id, "List", None, True, 10, False)
conn.add_software_parameter("forest_min_samples_split", software.id, "List", None, True, 11, False)
#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)

print "Software id is %d"%software.id
