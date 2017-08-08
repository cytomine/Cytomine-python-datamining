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


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine
import sys

#connect to cytomine : parameters to set
cytomine_host="localhost-core"
cytomine_public_key="0050f072-3896-4bef-ab30-2639470f2a3a"
cytomine_private_key="1a782b09-4c01-46f3-9ac7-61bb4f0c4c82"
id_project=5290
#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#Generic
execute_command = "python /software_router/algo/ldm_prediction/landmark_generic_predict.py " + \
                  "--cytomine_host $host " + \
                  "--cytomine_public_key $publicKey " + \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_predict_images $cytomine_predict_images " + \
                  "--model_load_from /software_router/algo/ldm_model_builder/ " + \
                  "--image_type jpg " + \
                  "--model_names $model_names " + \
                  "--verbose true"
software = conn.add_software("Landmark_Generic_Predictor_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("model_names",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, False, 3 , True)
conn.add_software_parameter("cytomine_id_software", software.id, "Number",0, True, 400, True)
conn.add_software_parameter("cytomine_id_project", software.id, "Number", 0, True, 500, True)
addSoftwareProject = conn.add_software_project(id_project,software.id)
print "Generic Prediction Software id is %d"%software.id

#DMBL
execute_command = "python /software_router/algo/ldm_prediction/landmark_dmbl_predict.py " + \
                  "--cytomine_host $host " + \
                  "--cytomine_public_key $publicKey " + \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_predict_images $cytomine_predict_images " + \
                  "--model_load_from /software_router/algo/ldm_model_builder/ " + \
                  "--image_type jpg " + \
                  "--model_name $model_name " + \
                  "--verbose true"             
software = conn.add_software("Landmark_DMBL_Predictor_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("model_name",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, True, 3 , True)
conn.add_software_parameter("cytomine_id_software", software.id, "Number",0, True, 400, True)
conn.add_software_parameter("cytomine_id_project", software.id, "Number", 0, True, 500, True)
addSoftwareProject = conn.add_software_project(id_project,software.id)
print "DMBL Prediction Software id is %d"%software.id


#LC
execute_command = "python /software_router/algo/ldm_prediction/landmark_lc_predict.py " + \
                  "--cytomine_host $host --cytomine_public_key $publicKey " + \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_predict_images $cytomine_predict_images " + \
                  "--model_load_from /software_router/algo/ldm_model_builder/ " + \
                  "--image_type jpg " + \
                  "--model_name $model_name " + \
                  "--verbose true"
software = conn.add_software("Landmark_LC_Predictor_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("model_name",             software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_predict_images",     software.id, "String", None, True, 2 , False)
conn.add_software_parameter("prediction_error",            software.id, "String", None, False, 3 , True)
conn.add_software_parameter("cytomine_id_software", software.id, "Number",0, True, 400, True)
conn.add_software_parameter("cytomine_id_project", software.id, "Number", 0, True, 500, True)
addSoftwareProject = conn.add_software_project(id_project,software.id)
print "LC Prediction Software id is %d"%software.id
