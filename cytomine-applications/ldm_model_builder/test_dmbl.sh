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
image_type=XXX #type of image, e.g 'jpg'
njobs=XXX #number of processors used for building the model, e.g 8
verbosity=XXX #verbosity, e.g 1
id_software=XXX #if of the model builder software, e.g 111251
model_save_path=XXX #path where the models will be stored e.g (/home/bob/models/)

#3. Edit model parameter values to build landmark detection model
# Note : you can specify several landmark and configuration in the array.
# model_name[i] will detect terms[i] with parameters R[i],...

NT_P1=X #Number of trees for phase 1
F_P1=X  #Number of pixel descriptors for phase 1
R_P1=X   #Radius for landmark pixel extraction during phase 1
sigma=X #Deviation of the gaussian
delta=X #Image reduction (]0,1])
P=X #Proportion of non-landmarks during phase 1
R_P2=X #Radius for pixel extraction during phase 2
ns_P2=X #Number of samples extracted per image during phase 2
NT_P2=X #Number of trees for phase 2
F_P2=X #Number of pixel descriptors for phase 2
filter_size=X #Filter size for phase 2
beta=X #Only pixels with beta*max_proba are kept after each iteration
n_iterations=X #Number of iterations for phase 3
n_candidates=X #Number of candidates for phase 3
sde=X #Standard deviation for phase 3
T=X #Number of edges for phase 3
id_terms=X #Term identifiers (eg. 122,321,143 or 'all')
trim=X #Training image identifiers (eg. 432,4324,5939 or 'all')

model_name=X #Name of the model (must be unique)

#4. Launch this script in a terminal (sh build_model.sh)

python build_dmbl_model.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_base_path $base_path --cytomine_id_software $id_software --cytomine_working_path $working_path --cytomine_training_images $trim --cytomine_id_project $id_project --image_type $image_type --model_njobs $njobs --cytomine_id_terms $id_terms --model_NT_P1 $NT_P1 --model_F_P1 $F_P1 --model_R_P1 $R_P1 --model_sigma $sigma --model_delta $delta --model_P $P --model_R_P2 $R_P2 --model_ns_P2 $ns_P2 --model_NT_P2 $NT_P2 --model_F_P2 $F_P2 --model_filter_size $filter_size --model_beta $beta --model_n_iterations $n_iterations --model_ncandidates $n_candidates --model_sde $sde --model_sde $T --model_save_to $model_save_path --model_name $model_name
