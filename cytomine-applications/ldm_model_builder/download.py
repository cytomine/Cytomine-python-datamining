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

import os
import shapely.wkt


def download_images(cytomine_connection, id_project):
	cytomine_connection.get_project_image_instances(id_project)
	cytomine_connection.dump_project_images(id_project=id_project, dest_path='/', max_size=True)


def download_annotations(cytomine_connection, id_project, working_dir):
	images = cytomine_connection.get_project_image_instances(id_project)
	images = images.data()
	xpos = {}
	ypos = {}
	terms = {}
	for image in images:
		annotations = cytomine_connection.get_annotations(id_project=id_project, showWKT=True, id_image=image.id)
		ann_data = annotations.data()
		for ann in ann_data:
			l = ann.location
			if l.rfind('POINT') == -1:
				pol = shapely.wkt.loads(l)
				poi = pol.centroid
			else:
				poi = shapely.wkt.loads(l)								
			(cx, cy) = poi.xy
			xpos[(ann.term[0], image.id)] = int(cx[0])
			ypos[(ann.term[0], image.id)] = image.height-int(cy[0])
			terms[ann.term[0]] = 1
	key_t = terms.keys()
	txt_path = working_dir+'%d/txt/' % id_project
	if not os.path.exists(txt_path):
		os.mkdir(txt_path)

	for image in images:
		F = open(txt_path+'%d.txt' % image.id, 'w')
		for t in key_t:
			if (t, image.id) in xpos:
				F.write('%d %d %d %f %f\n' % (t, xpos[(t, image.id)], ypos[(t, image.id)], xpos[(t, image.id)]/float(image.width), ypos[(t, image.id)]/float(image.height)))
		F.close()
	return xpos, ypos

