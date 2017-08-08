#!/usr/bin/env bash
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


#__author__          = "Vandaele Remy <remy.vandaele@ulg.ac.be>"
#__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"



#Example to run a landmark detection builder

#1. Adapt the add_software_builder.py to your needs and run it.

#2. Edit following XXX and 0 values with your cytomine identifiers
host='' #cytomine host, e.g "demo.cytomine.be"
public_key='' #your public key on the cytomine host
private_key='' #your private key on the cytomine host
base_path=XXX # cytomine base path e.g. /api/
working_path=XXX#cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=XXX #cytomine id project, e.g 35592
image_type='' #type of image, e.g 'jpg'
njobs=XXX #number of processors used for building the model, e.g 8
verbosity=XXX #verbosity, e.g 1
id_software=XXX #if of the model builder software, e.g 111251
model_save_path=XXX #path where the models will be stored e.g (/home/bob/models/)

#3. Edit model parameter values to build landmark detection model
# Note : you can specify several landmark and configuration in the array.
# model_name[i] will detect terms[i] with parameters R[i],...

terms=( )   #cytomine terms identifiers, e.g ( 35611,35619,35627 35635,35645 )
trim=( ) #training image identifiers, e.g ( 6102,7504,6809,3144 all )
R=( ) #max landmark distance, e.g ( 6 9 )
RMAX=( ) #max non-landmark distance, e.g ( 100 70 )
P=( ) #non-landmark proportion, e.g ( 2 3 )
npred=( ) #number of predictions per image, e.g ( 10000 5000 )
ntrees=( ) #number of trees, e.g ( 50 10 )
ntimes=( ) #number of rotations per image to add, e.g ( 1 2 )
angle=( ) #max angle of the rotation, e.g ( 0 5 )
depth=( ) #number of resolutions to use, e.g ( 5 3 )
step=( ) #landmarks are taken on a grid spaced by step, e.g (1 is minimum) ( 1 2 )
window_size=( ) #window size, e.g ( 8 8 )
feature_type=( ) #feature type "raw","sub","gaussian","haar", e.g ( 'sub' 'gaussian' )
feature_haar_n=( ) #number of haar-like pixel descriptors ( 1000 2000 )
feature_gaussian_n=( ) #number of gaussian pixel descriptors ( 2000 1000 )
feature_gaussian_std=( ) #standard deviation for gaussian pixel descriptor extraction
model_name=( ) #name of the models, e.g ( 'first_model' 'second_model' )

#4. Run this script in a terminal (sh build_model.sh)
for ((i=0;i<${#terms[@]};++i)); do
	python build_generic_model.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path $base_path --cytomine_id_software $id_software --cytomine_working_path $working_path --cytomine_training_images ${trim[i]} --model_feature_type ${feature_type[i]} --model_feature_haar_n ${feature_haar_n[i]} --model_feature_gaussian_n ${feature_gaussian_n[i]} --model_feature_gaussian_std ${feature_gaussian_std[i]} --cytomine_id_term ${terms[i]} --cytomine_id_project $id_project --image_type $image_type --model_njobs $njobs --model_R ${R[i]} --model_RMAX ${RMAX[i]} --model_P ${P[i]} --model_npred ${npred[i]} --model_ntrees ${ntrees[i]} --model_ntimes ${ntimes[i]} --model_angle ${angle[i]} --model_depth ${depth[i]} --model_step ${step[i]} --model_wsize ${window_size[i]} --model_save_to $model_save_path --model_name ${model_name[i]}
done

