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



#Example to perform a validation on one landmark
#Output format
#<IMAGE ID> <X DETECTED> <Y DETECTED> <X REAL> <Y REAL> <EUCLIDEAN ERRROR>
#...
#<TERM ID> <MIN CONF INTERVAL 95%> <MEAN> <MAX CONF INTERVAL 95%> 

#1. Edit following XXX and 0 values with your cytomine identifiers

id_software=XXX #Cytomine id software, e.g, on demo, 109777
host="" #cytomine host, e.g "demo.cytomine.be"
public_key=XXX #your public key on the cytomine host
private_key=XXX #your private key on the cytomine host
base_path=XXX # cytomine base path e.g. /api/
working_path=XXX #cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=0 #cytomine id project, e.g 35592
image_type=XXX #type of image, e.g 'jpg'
njobs=0 #number of processors used for building the model, e.g 8
verbosity=0 #verbosity
#2. Edit model parameters for the validation

term=0 #cytomine id term identifiers, e.g 35611 
R=0 #max landmark distance, e.g 6
RMAX=0 #max non-landmark distance, e.g 100
P=0 #non-landmark proportion, e.g 4
npred=0 #number of predictions per image, e.g 50000
ntrees=0 #number of trees per image, e.g 50
ntimes=0 #number of rotations per image to add, e.g 2
angle=0 #max angle of the rotation, e.g 30
depth=0 #number of resolutions to use, e.g 6
step=0 #landmarks are taken on a grid spaced by step, e.g 1
window_size=0 #window size, e.g 8

#3. Run this script in a terminal (sh build_model.sh)

python validation.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_id_software $id_software --cytomine_base_path $base_path --cytomine_working_path $working_path --cytomine_id_term $term --cytomine_id_project $id_project --image_type $image_type --model_njobs $njobs --model_R $R --model_RMAX $RMAX --model_P $P --model_npred $npred --model_ntrees $ntrees --model_ntimes $ntimes --model_angle $angle --model_depth $depth --model_step $step --model_wsize $window_size --verbose $verbosity


