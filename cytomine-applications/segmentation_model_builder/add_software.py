__author__ = 'stevben,rmaree'

import cytomine
import sys

#connect to cytomine

cytomine_host="XXX"
cytomine_public_key="XXX"
cytomine_private_key="XXX"
id_project=0

#Connection to Cytomine Core
conn = cytomine.Cytomine(cytomine_host, cytomine_public_key, cytomine_private_key, base_path = '/api/', working_path = '/tmp/', verbose= True)


#define software parameter template
software = conn.add_software("3Pyxit_SegmentationModel_Builder", "pyxitSuggestedTermJobService","ValidateAnnotation")
conn.add_software_parameter("pyxit_target_width", software.id, "Number", 16, True, 500, False)
conn.add_software_parameter("pyxit_target_height", software.id, "Number", 16, True, 600, False)
conn.add_software_parameter("pyxit_n_subwindows", software.id, "Number", 10, True, 200, False)
conn.add_software_parameter("pyxit_colorspace", software.id, "Number", 2, True, 900, False)
conn.add_software_parameter("pyxit_interpolation", software.id, "Number", 2, True, 700, False)
conn.add_software_parameter("pyxit_transpose", software.id, "Number", 0, True, 800, False)
conn.add_software_parameter("pyxit_fixed_size", software.id, "Boolean", "true", True, 905, False)
conn.add_software_parameter("forest_n_estimators", software.id, "Number", 10, True, 1100, False)
conn.add_software_parameter("forest_max_features", software.id, "Number", 1, True, 1200, False)
conn.add_software_parameter("forest_min_samples_split", software.id, "Number", 1, True, 1300, False)
conn.add_software_parameter("pyxit_save_to", software.id, "String", "/tmp", False, 20, False)
conn.add_software_parameter("pyxit_n_jobs", software.id, "Number", -1, True, 1000, False)

#add software to a given project
addSoftwareProject = conn.add_software_project(id_project,software.id)




