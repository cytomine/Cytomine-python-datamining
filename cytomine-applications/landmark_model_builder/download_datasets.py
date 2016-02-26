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

import sys
import cytomine
import os
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.affinity import *
import shapely.wkt

"""
Download all the images of the project with id id_project
The images will be saved in the working directory specified at the creation of 
the cytomine_connection object.
"""
def download_images(cytomine_connection, id_project):
	images = cytomine_connection.get_project_image_instances(id_project)
	images = images.data()
	ima = images.pop()
	image_size = max(ima.width,ima.height)
	cytomine_connection.dump_project_images(id_project=id_project, dest_path='/', max_size=True)

"""
Download all the landmark annotations of the images in the project with id
id_project
Only the terms for which the ids are specifieds in id_terms will be downloaded,
and written line by line in a file in the order of id_terms.

"""
def download_annotations(cytomine_connection, id_project, id_terms, working_dir):
	images = cytomine_connection.get_project_image_instances(id_project)
	images = images.data()
	xpos = {}
	ypos = {}
	terms = {}
	for image in images:
		annotations = cytomine_connection.get_annotations(id_project=id_project,showWKT=True,id_image=image.id)	
		ann_data = annotations.data()
		for ann in ann_data:
			l = ann.location
			if(l.rfind('POINT')==-1):
				pol = shapely.wkt.loads(l)
				poi = pol.centroid
			else:
				poi = shapely.wkt.loads(l)								
			(cx,cy) = poi.xy
			xpos[(ann.term[0],image.id)] = int(cx[0])
			ypos[(ann.term[0],image.id)] = image.height-int(cy[0])
			terms[ann.term[0]]=1
	key_t = terms.keys()
	txt_path = working_dir+'%d/txt/'%id_project
	if(not os.path.exists(txt_path)):
		os.mkdir(txt_path)

	for image in images:
		F = open(txt_path+'%d.txt'%(image.id),'w')
		for i in range(len(id_terms)):
			if((id_terms[i],image.id) in xpos):
				F.write('%d %d\n'%(xpos[(id_terms[i],image.id)],ypos[(id_terms[i],image.id)]))
		F.close()
	return xpos,ypos
        
if __name__ == "__main__":
	
	public_key = 'c3090afd-b793-4ae2-a5f3-2cfdaee01f32'
	private_key = 'd7768538-db66-4aa6-882c-55f520f1be69'
	wpath = sys.argv[1]
	if(wpath.endswith('/')):
		wpath = wpath+'/'
	cytomine_connection = cytomine.Cytomine('demo.cytomine.be',public_key,private_key,base_path='/api/',working_path=wpath,verbose=True)
	
	droso_terms = [6579647,6581077,6581992,6583116,6584107,6585116,6586002,6587114,6587962,6588763,6589668,6590562,6591526,6592482,6593390]
	cepha_terms = [6625929,6626956,6628031,6628982,6630085,6630930,6632153,6633169,6634164,6635158,6636231,6637186,6638098,6638869,6639680,6640638,6641592,6641602,6641610]
	zebra_terms = [6555577,6555589,6555603,6555613,6555621,6555631,6555631,6555647,6555657,6555665,6555675,6555681,6555691,6555699,6555709,6555717,6555727,6555735,6555745,6555753,6555761,6555769,6555777,6555787,6555795]
	droso_id_project = 6575282
	cepha_id_project = 6623446
	zebra_id_project = 6555554
	

	download_images(cytomine_connection,droso_id_project)
	download_annotations(cytomine_connection,droso_id_project,droso_terms,wpath)
	
	download_images(cytomine_connection,cepha_id_project)
	download_annotations(cytomine_connection,cepha_id_project,cepha_terms,wpath)
	
	download_images(cytomine_connection,zebra_id_project)
	download_annotations(cytomine_connection,zebra_id_project,zebra_terms,wpath)
