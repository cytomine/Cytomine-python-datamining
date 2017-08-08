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


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"

import cytomine

#connect to cytomine : parameters to set
cytomine_host="localhost-core"
cytomine_public_key="0050f072-3896-4bef-ab30-2639470f2a3a"
cytomine_private_key="1a782b09-4c01-46f3-9ac7-61bb4f0c4c82"
id_project=5290

conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)

"""
Generic Landmark model builder
"""
execute_command = "python /software_router/algo/ldm_model_builder/build_generic_model.py "+ \
                  "--cytomine_host $host "+ \
                  "--cytomine_public_key $publicKey "+ \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_id_terms $cytomine_id_terms " + \
                  "--cytomine_training_images $cytomine_training_images " + \
                  "--image_type jpg " + \
                  "--model_njobs $model_njobs " + \
                  "--model_R $model_R " + \
                  "--model_RMAX $model_RMAX " + \
                  "--model_P $model_P " + \
                  "--model_npred $model_npred " + \
                  "--model_ntrees $model_ntrees " + \
                  "--model_ntimes $model_ntimes " + \
                  "--model_angle $model_angle " + \
                  "--model_depth $model_depth " + \
                  "--model_step $model_step " + \
                  "--model_wsize $model_wsize " + \
                  "--model_feature_type $model_feature_type " + \
                  "--model_feature_haar_n $model_feature_haar_n " + \
                  "--model_feature_gaussian_n $model_feature_gaussian_n " + \
                  "--model_feature_gaussian_std $model_feature_gaussian_std " + \
                  "--model_save_to /software_router/algo/ldm_model_builder/ " + \
                  "--model_name $model_name " + \
                  "--verbose true"
software = conn.add_software("GLM_Builder_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("cytomine_id_terms",           software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_training_images",    software.id, "String", None, True, 2,  False)
conn.add_software_parameter("model_name",                  software.id, "String", None, True, 3,  False)
conn.add_software_parameter("model_njobs",                 software.id, "Number", None, True, 4 , False)
conn.add_software_parameter("model_R",                     software.id, "Number", None, True, 5 , False)
conn.add_software_parameter("model_RMAX",                  software.id, "Number", None, True, 6 , False)
conn.add_software_parameter("model_P",                     software.id, "Number", None, True, 7 , False)
conn.add_software_parameter("model_npred",                 software.id, "Number", None, True, 8 , False)
conn.add_software_parameter("model_ntimes",                software.id, "Number", None, True, 9 , False)
conn.add_software_parameter("model_angle",                 software.id, "Number", None, True, 10, False)
conn.add_software_parameter("model_depth",                 software.id, "Number", None, True, 11, False)
conn.add_software_parameter("model_wsize",                 software.id, "Number", None, True, 12, False)
conn.add_software_parameter("model_ntrees",                software.id, "Number", None, True, 13, False)
conn.add_software_parameter("model_step",                  software.id, "Number", None, True, 14, False)
conn.add_software_parameter("forest_max_features",         software.id, "Number", None, True, 15, False)
conn.add_software_parameter("forest_min_samples_split",    software.id, "Number", None, True, 16, False)
conn.add_software_parameter("model_feature_type",          software.id, "String", None, True, 17, False)
conn.add_software_parameter("model_feature_haar_n",        software.id, "Number", None, True, 18, False)
conn.add_software_parameter("model_feature_gaussian_n",    software.id, "Number", None, True, 19, False)
conn.add_software_parameter("model_feature_gaussian_std",  software.id, "Number", None, True, 20, False)
conn.add_software_parameter("cytomine_id_software",        software.id, "Number", 0,    True, 400, True)
conn.add_software_parameter("cytomine_id_project",         software.id, "Number", 0,    True, 500, True)
conn.add_software_project(id_project,software.id)
print "Generic Landmark Model Builder Software id is %d"%software.id

"""
DMBL
"""
execute_command = "python /software_router/algo/ldm_model_builder/build_dmbl_model.py "+ \
                  "--cytomine_host $host "+ \
                  "--cytomine_public_key $publicKey "+ \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_id_terms $cytomine_id_terms " + \
                  "--cytomine_training_images $cytomine_training_images " + \
                  "--model_name $model_name " + \
                  "--image_type jpg " + \
                  "--model_save_to /software_router/algo/ldm_model_builder/ " + \
                  "--model_NT_P1 $model_NT_P1 " + \
                  "--model_F_P1 $model_F_P1 " + \
                  "--model_R_P1 $model_R_P1 " + \
                  "--model_sigma $model_sigma " + \
                  "--model_delta $model_delta " + \
                  "--model_P $model_P " + \
                  "--model_R_P2 $model_R_P2 " + \
                  "--model_ns_P2 $model_ns_P2 " + \
                  "--model_NT_P2 $model_NT_P2 " + \
                  "--model_F_P2 $model_F_P2 " + \
                  "--model_filter_size $model_filter_size " + \
                  "--model_beta $model_beta " + \
                  "--model_n_iterations $model_n_iterations " + \
                  "--model_ncandidates $model_ncandidates " + \
                  "--model_sde $model_sde " + \
                  "--model_T $model_T " + \
                  "--verbose true"
software = conn.add_software("DMBL_Landmark_Model_Builder_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("cytomine_id_terms",                      software.id, "String", None, True, 1 , False)
conn.add_software_parameter("cytomine_training_images",               software.id, "String", None, True, 2 , False)
conn.add_software_parameter("model_name",                             software.id, "String", None, True, 3 , False)
conn.add_software_parameter("model_njobs",                     software.id, "Number", None, True, 4 , False)
conn.add_software_parameter("model_NT_P1",                     software.id, "Number", None, True, 5 , False)
conn.add_software_parameter("model_F_P1",                      software.id, "Number", None, True, 6 , False)
conn.add_software_parameter("model_R_P1",                      software.id, "Number", None, True, 7 , False)
conn.add_software_parameter("model_sigma",                     software.id, "Number", None, True, 8 , False)
conn.add_software_parameter("model_delta",                     software.id, "Number", None, True, 9 , False)
conn.add_software_parameter("model_P",                         software.id, "Number", None, True, 10 , False)
conn.add_software_parameter("model_R_P2",                      software.id, "Number", None, True, 11 , False)
conn.add_software_parameter("model_ns_P2",                     software.id, "Number", None, True, 12 , False)
conn.add_software_parameter("model_NT_P2",                     software.id, "Number", None, True, 13, False)
conn.add_software_parameter("model_F_P2",                      software.id, "Number", None, True, 14, False)
conn.add_software_parameter("model_filter_size",               software.id, "Number", None, True, 15, False)
conn.add_software_parameter("model_beta",                      software.id, "Number", None, True, 16, False)
conn.add_software_parameter("model_n_iterations",              software.id, "Number", None, True, 17, False)
conn.add_software_parameter("model_ncandidates",               software.id, "Number", None, True, 18, False)
conn.add_software_parameter("model_sde",                       software.id, "Number", None, True, 19, False)
conn.add_software_parameter("model_T",                         software.id, "Number", None, True, 20, False)
conn.add_software_parameter("cytomine_id_software",            software.id, "Number",0, True, 400, True)
conn.add_software_parameter("cytomine_id_project",             software.id, "Number", 0, True, 500, True)
conn.add_software_project(id_project,software.id)
print "DMBL Software id is %d"%software.id

"""
LC
"""
execute_command = "python /software_router/algo/ldm_model_builder/build_lc_model.py " + \
                  "--cytomine_host $host "+ \
                  "--cytomine_public_key $publicKey " + \
                  "--cytomine_private_key $privateKey " + \
                  "--cytomine_base_path /api/ " + \
                  "--cytomine_working_path /software_router/algo/ldm_model_builder/ " + \
                  "--cytomine_id_software $cytomine_id_software " + \
                  "--cytomine_id_project $cytomine_id_project " + \
                  "--cytomine_id_terms $cytomine_id_terms " + \
                  "--cytomine_training_images $cytomine_training_images " + \
                  "--image_type jpg " + \
                  "--model_njobs $model_njobs " + \
                  "--model_D_MAX $model_D_MAX " + \
                  "--model_n_samples $model_n_samples " + \
                  "--model_W $model_W " + \
                  "--model_n $model_n " + \
                  "--model_T $model_T " + \
                  "--model_step $model_step " + \
                  "--model_n_reduc $model_n_reduc " + \
                  "--model_R_MAX $model_R_MAX " + \
                  "--model_R_MIN $model_R_MIN " + \
                  "--model_alpha $model_alpha " + \
                  "--model_save_to /software_router/algo/ldm_model_builder/ " + \
                  "--model_name $model_name " + \
                  "--verbose true"
software = conn.add_software("LC_Landmark_Model_Builder_Pigeon", "createRabbitJobWithArgsService","ValidateAnnotation", execute_command)
conn.add_software_parameter("cytomine_id_terms",        software.id, "String", None, True, 1, False)
conn.add_software_parameter("cytomine_training_images", software.id, "String", None, True, 2, False)
conn.add_software_parameter("model_name",               software.id, "String", None, True, 3, False)
conn.add_software_parameter("model_njobs",       software.id, "Number", None, True, 4, False)
conn.add_software_parameter("model_D_MAX",       software.id, "Number", None, True, 5, False)
conn.add_software_parameter("model_n_samples",   software.id, "Number", None, True, 6, False)
conn.add_software_parameter("model_W",           software.id, "Number", None, True, 7, False)
conn.add_software_parameter("model_n",           software.id, "Number", None, True, 8, False)
conn.add_software_parameter("model_T",           software.id, "Number", None, True, 9, False)
conn.add_software_parameter("model_step",        software.id, "Number", None, True, 10, False)
conn.add_software_parameter("model_n_reduc",     software.id, "Number", None, True, 11, False)
conn.add_software_parameter("model_R_MAX",       software.id, "Number", None, True,12, False)
conn.add_software_parameter("model_R_MIN",       software.id, "Number", None, True,13, False)
conn.add_software_parameter("model_alpha",       software.id, "Number", None, True,14, False)
conn.add_software_parameter("cytomine_id_software", software.id, "Number",0, True, 400, True)
conn.add_software_parameter("cytomine_id_project", software.id, "Number", 0, True, 500, True)
conn.add_software_project(id_project,software.id)
print "LC Software id is %d"%software.id
