#!/usr/bin/env bash

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


#__author__          = "Mar�e Raphael <raphael.maree@ulg.ac.be>"
#__copyright__       = "Copyright 2010-2015 University of Li�ge, Belgium, http://www.cytomine.be/"


# Pyxit Classification Model building using algorithms in Maree et al., TR 2014

#1. Edit add_software.py and add the software to your Cytomine project if not existing yet


#2. Edit following XXX and 0 values with your cytomine identifiers
cytomine_host="beta.cytomine.be"
cytomine_id_software=179703698
cytomine_public_key="ad014190-2fba-45de-a09f-8665f803ee0b"
cytomine_private_key="767512dd-e66f-4d3c-bb46-306fa413a5eb"
cytomine_id_project=716498
cytomine_working_path=$HOME/downloaded
cytomine_annotation_projects=716498
cytomine_predict_terms=676390 #id of terms to be grouped into the positive class (e.g. tumor) separated by ,
cytomine_excluded_terms=675999,676026,676176,933004,676407,8844862,8844845,9444456,15054705,15054765,15109451,15109483,15109489,15109495,22042230,28792193,30559888 #id of terms that will not be used (neither positive nor negative class) separated by ,
model_file=cell_model.pkl
zoom=1 #zoom_level at which annotations are dumped


#2. Edit pyxit parameter values to build models (see Maree et al. Technical Report 2014)
windowsize=16 #resized_size for subwindows
colorspace=2 #colorspace to encode pixel values
njobs=4 #number of parallel threads
interpolation=1 #interpolation method to rescale subwindows to fixed size
nbt=10 #numer of extra-trees
k=28 #tree node filterning parameter (number of tests) in extra-trees
nmin=10 #minimum node sample size in extra-trees
subw=10 #number of extracted subwindows per annotation crop image
min_size=0.1 #minimum size of subwindows (proportionnaly to image size: 0.1 means minimum size is 10% of min(width,height) of original image
max_size=0.9 #maximum size of subwindows (...)


#4. Run
#Note:
#-Annotations will be dumped into $working_path/$id_project/zoom_level/$zoom/...
#-Model will be created into $working_path/models/$model_file
#-If you want to use only reviewed annotations, uncomment --cytomine_reviewed
#-If you want to build a model better robust to orientation, uncomment pyxit_transpose (which applies random right-angle rotation to subwindows)
#But it the orientation is related to object classes (e.g. 6 and 9 in digit classification) do not use transpose
#-Code could be modified to specify other filters (e.g. user ids, image ids, increasedarea)


python add_and_run_job.py --cytomine_host $cytomine_host \
            --cytomine_public_key $cytomine_public_key \
            --cytomine_private_key $cytomine_private_key \
            --cytomine_base_path /api/ \
            --cytomine_id_software $cytomine_id_software \
            --cytomine_working_path $cytomine_working_path/annotations/ \
            --cytomine_id_project $cytomine_id_project \
            -z $zoom \
            --cytomine_annotation_projects $cytomine_annotation_projects \
            --cytomine_excluded_terms $cytomine_excluded_terms \
            --pyxit_target_width $windowsize \
            --pyxit_target_height $windowsize \
            --pyxit_colorspace $colorspace --pyxit_n_jobs $njobs \
            --pyxit_min_size $min_size \
            --pyxit_max_size $max_size --pyxit_interpolation $interpolation \
            --forest_n_estimators $nbt \
            --forest_max_features $k --forest_min_samples_split $nmin \
            --pyxit_n_subwindows $subw \
            --svm 1 --pyxit_save_to $HOME/models/$model_file \
            --cytomine_dump_type 1 \
            --pyxit_transpose false --pyxit_fixed_size false --cytomine_reviewed false \
            --verbose false


# Error =>
#  1) passing only pyxit_parameters instead of all_parameters
#  2) dir_ls not set
