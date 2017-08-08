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
working_path=XXX #cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=XXX #cytomine id project, e.g 35592
image_type='' #type of image, e.g 'jpg'
njobs=XXX #number of processors used for building the model, e.g 8
verbosity=XXX #verbosity, e.g 1
id_software=XXX #if of the model builder software, e.g 111251
model_save_path=XXX #path where the models will be stored e.g (/home/bob/models/)

#3. Edit model parameter values to build landmark detection model

D_MAX=XX #max distance to the landmarks for pixel extraction
n_samples=XX #number of pixel samples to extract (per image)
W=XX #size of the window for pixel extraction
n=XX #number of haar-like features for pixel description
T=XX #number of trees
step=XX #step for landmark prediction
n_reduc=XX #Reduction for pca
R_MAX=XX #Max radius to search 
R_MIN=XX #Min radius
alpha=XX #Radius is actualized with alpha
id_terms=XX #Terms to be considered. 6579647,6588763,6581077,6584107 or all
trim=XX #Identifiers of the training images
model_name='' #name of the model

#4. Run this script in a terminal (sh build_model.sh)

python build_lc_model.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path $base_path --cytomine_id_software $id_software --cytomine_working_path $working_path --cytomine_id_project $id_project --cytomine_training_images $trim --image_type $image_type --model_njobs $njobs --cytomine_id_terms $id_terms --model_D_MAX $D_MAX --model_n_samples $n_samples --model_W $W --model_n $n --model_T $T --model_step $step --model_n_reduc $n_reduc --model_R_MAX $R_MAX --model_R_MIN $R_MIN --model_alpha $alpha --model_save_to $model_save_path --model_name $model_name
