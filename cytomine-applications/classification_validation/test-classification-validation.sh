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

# Author: "Marée Raphaël <raphael.maree@ulg.ac.be>"
# Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/


#Example to run a cross-validation of classification model using Cytomine annotations

#0. Edit the add_software.py file and run python add_software.py to add the software to Cytomine Core and project (once)

#1. Edit following XXX with your cytomine identifiers and other parameter values
cytomine_host="XXX"
cytomine_public_key="XXX" #if human user then creates a new userjob, otherwise use provided userjob keys
cytomine_private_key="XXX" 
software=XXX
id_project=XXX  #project where to run the job
annotation_projects=XXX  #projects from which to download annotations to build the models (should include the id_project)
working_path=/bigdata/tmp/cytomine/
zoom=1  #zoom_level for cropped annotations
excluded_terms=XXX #annotations from these terms will not be downloaded id_term1,id_term2,...


#2. Edit pyxit parameter values to build models (see Maree et al. Technical Report 2014)
windowsize=16 #resized_size for subwindows 
colorspace=2 #colorspace to encode pixel values
njobs=10 #number of parallel threads
interpolation=1 #interpolation method to rescale subwindows to fixed size
nbt=10 #numer of extra-trees
k=28 #tree node filterning parameter (number of tests) in extra-trees
nmin=10 #minimum node sample size in extra-trees
subw=100 #number of extracted subwindows per annotation crop image
min_size=0.1 #minimum size of subwindows (proportionnaly to image size: 0.1 means minimum size is 10% of min(width,height) of original image
max_size=0.9 #maximum size of subwindows (...)

#3. Edit Cross-validation parameters
cv_k_folds=3 #number of folds for cross-validation


#4. Run
# Note: 
# 1) This script dump annotations from annotation projects, extract random subwindows from annotation images, then build classification models.
# Models are evaluated by cross-validation (see cv_k_folds parameters...). Confusion matrices are printed on local output.
# At the end the script uploads all predictions to Cytomine core server so confusion matrices can be seen through the web client in the "Jobs"
# tab of the id_project.
# 2) Read our paper Maree et al., 2014 for more information about recommended parameter values
# 3) svm 1 means ET-FL mode with LIBLINEAR (see code and paper).
python add_and_run_job.py --cytomine_host $cytomine_host --cytomine_public_key $cytomine_public_key --cytomine_private_key $cytomine_private_key --cytomine_base_path /api/ --cytomine_id_software $software --cytomine_working_path $working_path/annotations/ --cytomine_id_project $id_project --cytomine_annotation_projects $annotation_projects -z $zoom --cytomine_excluded_terms $excluded_terms --pyxit_target_width $windowsize --pyxit_target_height $windowsize --pyxit_colorspace $colorspace --pyxit_n_jobs $njobs --pyxit_transpose --pyxit_min_size $min_size --pyxit_max_size $max_size --pyxit_interpolation $interpolation --forest_n_estimators $nbt --forest_max_features $k --forest_min_samples_split $nmin --pyxit_n_subwindows $subw --cytomine_dump_type 1 --svm 1 --pyxit_save_to $working_path/models/model.pkl --cv_k_folds $cv_k_folds --verbose 1 --cytomine_fixed_tile false --cytomine_n_shifts 0 --cytomine_reviewed false --pyxit_fixed_size false --cv_shuffle false

