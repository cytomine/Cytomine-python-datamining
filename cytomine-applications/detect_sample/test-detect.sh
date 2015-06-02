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


#Example to run a tissue section detector in all whole gigapixel images of a project

#0. Edit the add_software.py file to add the software to Cytomine Core (once) and project (once)

#1. Edit following XXX and 0 values with your cytomine identifiers

#2. Replace XXX values by your settings
cytomine_host="XXX"
cytomine_public_key="XXX"  #if human user then creates a new userjob, otherwise use provided userjob keys
cytomine_private_key="XXX"
cytomine_id_project=XXX
cytomine_id_software=XXX
cytomine_predict_term=XXX #id of ontology term to associate to objects detected by detector
cytomine_working_path=/bigdata/tmp/cytomine/


cytomine_max_image_size=1000 
cytomine_erode_iterations=3
cytomine_dilate_iterations=3
cytomine_athreshold_blocksize=551
cytomine_athreshold_constant=5


#Note: 
#This script downloads thumbnails of whole-slide image, applies adaptive thresholding, and uploads geometries to Cytomine-Core as userjob layer
python ../detect_sample.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_working_path $cytomine_working_path --cytomine_id_software $cytomine_id_software --cytomine_id_project $cytomine_id_project --cytomine_predict_term $cytomine_predict_term --cytomine_max_image_size $cytomine_max_image_size --cytomine_erode_iterations $cytomine_erode_iterations --cytomine_dilate_iterations $cytomine_dilate_iterations --cytomine_athreshold_blocksize $cytomine_athreshold_blocksize --cytomine_athreshold_constant $cytomine_athreshold_constant --verbose 1


