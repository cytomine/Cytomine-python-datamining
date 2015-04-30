#!/bin/bash

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


#__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
#__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


#Example to run a sample (e.g. tissue section) detection in whole gigapixel images of a project

#0. Edit the add_software.py file to add the software to Cytomine Core (once) and project (once)

#1. Edit following XXX and 0 values with your cytomine identifiers

#2. Replace XXX values by your settings
cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
cytomine_id_project=XXX
cytomine_id_image=XXX
cytomine_id_software=XXX
cytomine_predict_term=XXX #id of term to associate to objects detected by object finder (0 if undefined)
cytomine_working_path=/bigdata/tmp/cytomine/
cytomine_zoom_level=3 #zoom level


python detect-sample.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_id_software $cytomine_id_software --cytomine_working_path /data/home/maree/tmp/cytomine/ --cytomine_id_project $id_project --cytomine_predict_term $cytomine_predict_term





