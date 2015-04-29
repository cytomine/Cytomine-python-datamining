#!/bin/bash

# ---------------------------------------------------------------------------------------------------------
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


# Pyxit Classification Model Prediction using algorithms in Maree et al., TR 2014

#0. Edit the add_software.py file to add the software to Cytomine Core and project (once)

#1. Edit following XXX and 0 values with your cytomine identifiers
host="XXX"
cytomine_id_software=XXX #identifier of the cytomine classification_prediction software (added by add_software)
public_key="XXX"
private_key="XXX"
cytomine_id_project=XXX #identifier of the project in which to work
cytomine_working_path=/bigdata/tmp/cytomine/
model_dir="models/"
model_file=classification_model.pkl
pyxit_save_to=$cytomine_working_path$model_dir$model_file  #local path to the model (built using code in classification_model_builder/)
cytomine_id_userjob=XXX  #identifier of the userjob (or regular user) from which to retrieve annotations to classify

#2. Edit pyxit parameter values to apply segmentation model (most parameter values are derived from the model file, in theory should keep the same values)
cytomine_zoom_level=2 #zoom level to extract annotations (0 = maximum resolution)
cytomine_dump_type=1 #original crop image of the annotation (2 = with alpha mask)


#Notes:
#- This script will apply a classifier to a set of annotations within an image retrieved from Cytomine-Core. 
# Usually, these annotations were previously produced by another job (e.g. a object finder using thresholding).
# It will upload new annotations (in the userjob layer) with their predicted terms.

for cytomine_id_image in XXX #identifier of the image from which annotations are retrieved
do
    python ../add_and_run_job.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path /api/ --cytomine_id_software $cytomine_id_software --cytomine_working_path $cytomine_working_path --cytomine_id_project $cytomine_id_project --cytomine_id_image $cytomine_id_image --cytomine_zoom_level $cytomine_zoom_level --cytomine_id_userjob $cytomine_id_userjob --pyxit_save_to $pyxit_save_to --cytomine_dump_type $cytomine_dump_type
done


