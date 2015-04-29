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

#Edit parameters to connect to Cytomine-Core
cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=XXX

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)

software = conn.add_software("Classification_Prediction", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 20, False)
conn.add_software_parameter("cytomine_zoom_level", software.id, "Number", 0, True, 100,False)
conn.add_software_parameter("cytomine_dump_type", software.id, "Number", 1, True, 200,False)
conn.add_software_parameter("cytomine_id_userjob", software.id, "Number", 1, True, 300,False)



#Link software with project(s):
addSoftwareProject = conn.add_software_project(id_project,software.id)

