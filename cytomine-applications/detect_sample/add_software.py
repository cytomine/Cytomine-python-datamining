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


__author__          = "Maree Raphael <raphael.maree@ulg.ac.be>"
__copyright__       = "Copyright 2010-2015 University of Liege, Belgium, http://www.cytomine.be/"


import cytomine
import sys

#Connect to cytomine, edit connection values
cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=XXX


#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#define software parameter template
software = conn.add_software("Detect_Sample", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("cytomine_max_image_size", software.id, "Number", 0, True, 10, False)
conn.add_software_parameter("cytomine_erode_iterations", software.id, "Number", 0, True, 30, False)
conn.add_software_parameter("cytomine_dilate_iterations", software.id, "Number", 0, True, 40, False)
conn.add_software_parameter("cytomine_athreshold_constant", software.id, "Number", 0, True, 50, False)
conn.add_software_parameter("cytomine_athreshold_blocksize", software.id, "Number", 0, True, 60, False)


#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)
