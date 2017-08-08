# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
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
#__copyright__       = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"



#Example to run a landmark detection builder

#1. Adapt the add_software_builder.py to your needs and run it.

#2. Edit following XXX and 0 values with your cytomine identifiers
host='localhost-core' #cytomine host, e.g "demo.cytomine.be"
public_key='0ab78d51-3a6e-40e1-9b1d-d42c28bc1923' #your public key on the cytomine host
private_key='817d2e30-b4df-41d2-bb4b-fb29910b1d4e' #your private key on the cytomine host
base_path=/api/ # cytomine base path e.g. /api/
working_path=/home/remy/cytomine_working/ #cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=5290 #cytomine id project, e.g 35592
image_type='jpg' #type of image, e.g 'jpg'
njobs=4 #number of processors used for building the model, e.g 8
verbosity=0 #verbosity, e.g 1
id_software=168184 #if of the model builder software, e.g 111251
model_save_path=/home/remy/cytomine_models/ #path where the models will be stored e.g (/home/bob/models/)

#3. Edit model parameter values to build landmark detection model
# Note : you can specify several landmark and configuration in the array.
# model_name[i] will detect terms[i] with parameters R[i],...
D_MAX=10 #max landmark distance ( 6 9 6 7 7 )
n_samples=25 #max non-landmark distance, e.g ( 100 70 100 200 120 )
W=3 #non-landmark proportion, e.g ( 4 4 6 6 3 )
n=20 #number of predictions per image, e.g ( 50000 60000 20000 12000 21000 )
T=10 #number of trees per image, e.g ( 50 10 15 20 21 )
step=4 #number of rotations per image to add, e.g ( 1 2 1 2 1 )
n_reduc=20
R_MAX=100
R_MIN=1
alpha=0.5
id_terms=all #6579647,6588763,6581077,6584107
trim='all'

#angle=( 0 0 ) #max angle of the rotation, e.g ( 0 10 0 20 0 )
#depth=( 5 5 ) #number of resolutions to use, e.g ( 6 6 6 6 6 )
#step=( 2 2 ) #landmarks are taken on a grid spaced by step, e.g (1 is minimum) ( 1 1 2 2 1 )
#window_size=( 8 8 ) #window size, e.g ( 8 8 8 8 8 )
#feature_type=( 'gaussian' 'haar' )
#feature_haar_n=( 1600 1600 )
#feature_gaussian_n=( 1600 1600 )
#feature_gaussian_std=( 20 20 )
model_name='lcmod' #name of the models, e.g ( 'first_model' 'second_model' 'third_model' 'fourth_model' 'fifth_model' )

#4. Run this script in a terminal (sh build_model.sh)

python build_lc_model.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path $base_path --cytomine_id_software $id_software --cytomine_working_path $working_path --cytomine_id_project $id_project --cytomine_training_images $trim --image_type $image_type --model_njobs $njobs --cytomine_id_terms $id_terms --model_D_MAX $D_MAX --model_n_samples $n_samples --model_W $W --model_n $n --model_T $T --model_step $step --model_n_reduc $n_reduc --model_R_MAX $R_MAX --model_R_MIN $R_MIN --model_alpha $alpha --model_save_to $model_save_path --model_name $model_name
