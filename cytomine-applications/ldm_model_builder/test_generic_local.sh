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
host='localhost-core' #cytomine host, e.g "demo.cytomine.be"

public_key='0050f072-3896-4bef-ab30-2639470f2a3a' #your public key on the cytomine host
private_key='1a782b09-4c01-46f3-9ac7-61bb4f0c4c82' #your private key on the cytomine host
base_path=/api/ # cytomine base path e.g. /api/
working_path=/home/remy/cytomine_working/ #cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=5290 #cytomine id project, e.g 35592
image_type='jpg' #type of image, e.g 'jpg'
njobs=1 #number of processors used for building the model, e.g 8
verbosity=1 #verbosity, e.g 1
id_software=166417 #if of the model builder software, e.g 111251
model_save_path=/home/remy/cytomine_models/ #path where the models will be stored e.g (/home/bob/models/)

#3. Edit model parameter values to build landmark detection model
# Note : you can specify several landmark and configuration in the array.
# model_name[i] will detect terms[i] with parameters R[i],...

terms=( 10879 )   #cytomine terms identifiers, e.g ( 35611 35619 35627 35635 35645 )
trim=( 'all' )
R=( 6 ) #max landmark distance ( 6 9 6 7 7 )
RMAX=( 100 ) #max non-landmark distance, e.g ( 100 70 100 200 120 )
P=( 2 ) #non-landmark proportion, e.g ( 4 4 6 6 3 )
npred=( 10000 ) #number of predictions per image, e.g ( 50000 60000 20000 12000 21000 )
ntrees=( 10 ) #number of trees per image, e.g ( 50 10 15 20 21 )
ntimes=( 1 ) #number of rotations per image to add, e.g ( 1 2 1 2 1 )
angle=( 0 ) #max angle of the rotation, e.g ( 0 10 0 20 0 )
depth=( 5 ) #number of resolutions to use, e.g ( 6 6 6 6 6 )
step=( 2 ) #landmarks are taken on a grid spaced by step, e.g (1 is minimum) ( 1 1 2 2 1 )
window_size=( 8 ) #window size, e.g ( 8 8 8 8 8 )
feature_type=( 'haar' )
feature_haar_n=( 1600 )
feature_gaussian_n=( 1600 )
feature_gaussian_std=( 20 )
model_name=( 'first_model' ) #name of the models, e.g ( 'first_model' 'second_model' 'third_model' 'fourth_model' 'fifth_model' )

#4. Run this script in a terminal (sh build_model.sh)
for ((i=0;i<${#terms[@]};++i)); do
	python build_generic_model.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path $base_path --cytomine_id_software $id_software --cytomine_working_path $working_path --cytomine_training_images ${trim[i]} --model_feature_type ${feature_type[i]} --model_feature_haar_n ${feature_haar_n[i]} --model_feature_gaussian_n ${feature_gaussian_n[i]} --model_feature_gaussian_std ${feature_gaussian_std[i]} --cytomine_id_term ${terms[i]} --cytomine_id_project $id_project --image_type $image_type --model_njobs $njobs --model_R ${R[i]} --model_RMAX ${RMAX[i]} --model_P ${P[i]} --model_npred ${npred[i]} --model_ntrees ${ntrees[i]} --model_ntimes ${ntimes[i]} --model_angle ${angle[i]} --model_depth ${depth[i]} --model_step ${step[i]} --model_wsize ${window_size[i]} --model_save_to $model_save_path --model_name ${model_name[i]}
done

