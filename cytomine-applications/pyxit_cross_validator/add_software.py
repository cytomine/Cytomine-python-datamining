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


__author__          = "Mormont Romain <r.mormont@ulg.ac.be>"
__contributors__    = []
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"


import cytomine

#connect to cytomine
cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=XXX

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)

software = conn.add_software("Pyxit_Cross_Validator", "pyxitSuggestedTermJobService", "ValidateAnnotation")

conn.add_software_parameter("dir_ls", software.id, "String", "/tmp", False, 10, False)
conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 20, False)
conn.add_software_parameter("pyxit_n_subwindows", software.id, "Number", 10, True, 200,False)
conn.add_software_parameter("pyxit_min_size", software.id, "Number", 0.1, True, 300,False)
conn.add_software_parameter("pyxit_max_size", software.id, "Number", 1, True, 400,False)
conn.add_software_parameter("pyxit_target_width", software.id, "Number", 16, True, 500,False)
conn.add_software_parameter("pyxit_target_height", software.id, "Number", 16, True, 600,False)
conn.add_software_parameter("pyxit_interpolation", software.id, "Number", 2, True, 700,False)
conn.add_software_parameter("pyxit_transpose", software.id, "Number", 0, True, 800,False)
conn.add_software_parameter("pyxit_colorspace", software.id, "Number", 2, True, 900,False)
conn.add_software_parameter("pyxit_fixed_size", software.id, "Boolean", "false", True, 950, False)
conn.add_software_parameter("pyxit_n_jobs", software.id, "Number", -1, True, 1000,False)
conn.add_software_parameter("forest_n_estimators", software.id, "Number", 10, True, 1100,False)
conn.add_software_parameter("forest_max_features", software.id, "Number", 1, True, 1200,False)
conn.add_software_parameter("forest_min_samples_split", software.id, "Number", 1, True, 1300,False)
conn.add_software_parameter("svm", software.id, "Number", 0, False, 1600,False)
conn.add_software_parameter("svm_c", software.id, "Number", 1.0, False, 1700,False)

# Generic cytomine parameters
conn.add_software_parameter("cytomine_host", software.id, "String", default="")
conn.add_software_parameter("cytomine_public_key", software.id, "String", default="")
conn.add_software_parameter("cytomine_private_key", software.id, "String", default="")
conn.add_software_parameter("cytomine_base_path", software.id, "String", default="/api/")
conn.add_software_parameter("cytomine_working_path", software.id, "String", default="/tmp/")
conn.add_software_parameter("cytomine_zoom_level", software.id, "Number", default=0)
conn.add_software_parameter("cytomine_id_project", software.id, "Number")
conn.add_software_parameter("cytomine_id_software", software.id, "Number")

# Filtering images, annotations and terms?
conn.add_software_parameter("cytomine_users", software.id, "String", users)
conn.add_software_parameter("cytomine_excluded_terms", software.id, "String", default="")
conn.add_software_parameter("cytomine_excluded_annotations", software.id, "String", default="")
conn.add_software_parameter("cytomine_excluded_images", software.id, "String", default="")

# Include reviewed ?
conn.add_software_parameter("cytomine_include_reviewed", software.id, "String", default="False")
conn.add_software_parameter("cytomine_reviewed_users", software.id, "String", default="")
conn.add_software_parameter("cytomine_reviewed_images", software.id, "String", default="")

# Binary mapping
conn.add_software_parameter("cytomine_binary", software.id, "String", default="False")
conn.add_software_parameter("cytomine_positive_terms", software.id, "String")
conn.add_software_parameter("cytomine_negative_terms", software.id, "String")

# Ternary mapping
conn.add_software_parameter("cytomine_ternary", software.id, "String", default="False")
conn.add_software_parameter("cytomine_group1", software.id, "String")
conn.add_software_parameter("cytomine_group2", software.id, "String")
conn.add_software_parameter("cytomine_group3", software.id, "String")

# Images in the test set ?
conn.add_software_parameter("cytomine_test_images", software.id, "String")

# Method parameters
# Extra-trees
conn.add_software_parameter("forest_n_estimators", software.id, "Number", default=10)
conn.add_software_parameter("forest_min_samples_split", software.id, "String", default="1")
conn.add_software_parameter("forest_max_features", software.id, "String", default="16")

# Pyxit
conn.add_software_parameter("pyxit_tune_by_cv", software.id, "String", default="False")
conn.add_software_parameter("pyxit_target_width", software.id, "Number", default=16)
conn.add_software_parameter("pyxit_target_height", software.id, "Number", default=16)
conn.add_software_parameter("pyxit_fixed_size", software.id, "String", default="False")
conn.add_software_parameter("pyxit_n_subwindows", software.id, "Number", default=10)
conn.add_software_parameter("pyxit_transpose", software.id, "String", default="False")
conn.add_software_parameter("pyxit_interpolation", software.id, "Number", default=2)
conn.add_software_parameter("pyxit_colorspace", software.id, "String", default="2")
conn.add_software_parameter("pyxit_max_size", software.id, "String", default="0.1")
conn.add_software_parameter("pyxit_min_size", software.id, "String", default="0.9")

# Using ET-FL instead of ET-DIC
conn.add_software_parameter("svm", software.id, "String", default="False")
conn.add_software_parameter("svm_c", software.id, "String", default="1.0")

# Miscellaneous
conn.add_software_parameter("cv_images_out", software.id, "Number", default=1)
conn.add_software_parameter("pyxit_dir_ls", software.id, "String", default="/tmp/ls")
conn.add_software_parameter("n_jobs", software.id, "Number", default=1)
conn.add_software_parameter("pyxit_save_to", software.id, "String", default=None)
conn.add_software_parameter("verbose", software.id, "String", default="False")


#Link software with project:
addSoftwareProject = conn.add_software_project(id_project,software.id)
