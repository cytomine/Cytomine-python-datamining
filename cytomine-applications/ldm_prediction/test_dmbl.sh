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

#Example to run a landmark detection model to predict the position of landmarks in new images

#1. Edit add_software.py and add the software to your Cytomine project if not existing yet

#2. Edit these parameters.
id_software=XXX #Cytomine id software, e.g, on demo, 90687
host='' #cytomine host, e.g "demo.cytomine.be"
public_key='' #your public key on the cytomine host
private_key='' #your private key on the cytomine host
base_path=XXX # cytomine base path e.g. /api/
working_path=XXX #cytomine working path, e.g. /bigdata/tmp/cytomine/
id_project=XXX #cytomine id project
image_type='' #type of image, e.g 'jpg'
model_path=XXX #path to the location of the models (e.g /home/bob/models/)
model_names=''  #name of the models built with the model builder, e.g 'first_model,second_model,third_model,fourth_model,fifth_model'
verbosity=XXX #verbosity of the cytomine client, e.g 0
image_predict=''

#3. Run this script (sh predict.sh)

python landmark_dmbl_predict.py --cytomine_host $host --cytomine_public_key $public_key --cytomine_private_key $private_key --cytomine_id_software $id_software --cytomine_base_path $base_path --cytomine_working_path $working_path --cytomine_id_project $id_project --cytomine_predict_images $image_predict --model_load_from $model_path --model_name $model_name --image_type $image_type --verbose $verbosity