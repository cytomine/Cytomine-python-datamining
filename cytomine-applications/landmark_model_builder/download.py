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

import getopt
import sys
import cytomine
import os, optparse
import re
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.affinity import *
import shapely.wkt

def download_images(cytomine_connection, id_project):
	images = cytomine_connection.get_project_image_instances(id_project)
	images = images.data()
	ima = images.pop()
	image_size = max(ima.width,ima.height)
	cytomine_connection.dump_project_images(id_project=id_project, dest_path='/', max_size=True)

def download_annotations(cytomine_connection, id_project, working_dir):
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
		for t in key_t:
			if((t,image.id) in xpos):
				F.write('%d %d %d\n'%(t,xpos[(t,image.id)],ypos[(t,image.id)]))
		F.close()
	return xpos,ypos

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
        
if __name__ == "__main__":
	
	parameters = {
		'cytomine_host' : None,
		'cytomine_public_key' : '',
		'cytomine_private_key' : '',
		'cytomine_base_path' : '',
		'cytomine_working_path' : '',
		'cytomine_id_project' : None,
		'verbose':False
	}
	
	#main allows to download and store the data and the user coordinates in an offline repository
	
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Image downloading',prog='Cytomine Landmark Detector : Image downloader',version='0.1')
	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = 'XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = 'YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")
	
	options, arguments = p.parse_args( args = sys.argv)
	
	parameters['cytomine_host'] = options.cytomine_host	
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path	
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['verbose']=str2bool(options.verbose)
	
	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'],parameters['cytomine_public_key'],parameters['cytomine_private_key'],base_path=parameters['cytomine_base_path'],working_path=parameters['cytomine_working_path'],verbose=parameters['verbose'])
	download_images(cytomine_connection,parameters['cytomine_id_project'])
	download_annotations(cytomine_connection,parameters['cytomine_id_project'],parameters['cytomine_working_path'])
