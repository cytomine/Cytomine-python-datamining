# -*- coding: utf-8 -*-


#
# * Copyright (c) 2009-2015. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *	  http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# 


__author__		  = "Vandaele Rémy <remy.vandaele@ulg.ac.be>" 
__contributors__	= ["Hoyoux Renaud <renaud.hoyoux@ulg.ac.be>","Marée Raphaël <raphael.maree@ulg.ac.be>"]				
__copyright__	   = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"


import getopt
import sys
import cytomine
import os, optparse
import re
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from shapely.affinity import *
import shapely.wkt

#A software to export reviewed landmark coordinates for a project's images. 
#A line = one image, columns are landmark terms with x,y absolute positions. 
def main(argv):


	parser = optparse.OptionParser(description='Cytomine Datamining',
							  prog='cytomining',
							  version='cytomining 0.1')
	parser.add_option('--cytomine_host', dest='cytomine_host', help='cytomine_host')
	parser.add_option('--cytomine_public_key', dest='cytomine_public_key', help='cytomine_public_key')
	parser.add_option('--cytomine_private_key', dest='cytomine_private_key', help='cytomine_private_key')
	parser.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
	parser.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
	parser.add_option('--cytomine_base_path', dest='cytomine_base_path', help='cytomine base path')
	parser.add_option('--cytomine_working_path', dest='cytomine_working_path', help='cytomine_working_path base path')

	options, arguments = parser.parse_args( args = argv)


	#copy options
	parameters = {}
	parameters['cytomine_host'] = options.cytomine_host
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path	

	print "Connection to Cytomine server"
	cytomine_connection = cytomine.Cytomine(parameters["cytomine_host"], 
			parameters["cytomine_public_key"], 
			parameters["cytomine_private_key"] , 
			base_path = parameters['cytomine_base_path'], 
			working_path = parameters['cytomine_working_path'])

	#Create a new userjob if connected as human user
	current_user = cytomine_connection.get_current_user()
	if current_user.algo==False:
		print "adduserJob..."
		user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
		print "set_credentials..."
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
		print "done"
	else:
		user_job = current_user
		print "Already running as userjob"


	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status = job.RUNNING, status_comment = "Fetching data", progress = 0)

	images = cytomine_connection.get_project_image_instances(parameters['cytomine_id_project'])
	images = images.data()
	xpos = {}
	ypos = {}
	terms = {}

	progress = 0
	delta = 80/len(images)

	for image in images:

		progress += delta
		job = cytomine_connection.update_job_status(job, status = job.RUNNING, status_comment = "Fetching data", progress = progress)

		annotations = cytomine_connection.get_annotations(id_project=parameters['cytomine_id_project'],showWKT=True,id_image=image.id, reviewed_only=True)	
		ann_data = annotations.data()
		for ann in ann_data:
			if(len(ann.term) > 0):
				l = ann.location
				if(l.rfind('POINT')==-1):
					pol = shapely.wkt.loads(l)
					poi = pol.centroid
				else:
					poi = shapely.wkt.loads(l)								
				(cx,cy) = poi.xy
				xpos[(ann.term[0],image.id)] = int(cx[0])
				ypos[(ann.term[0],image.id)] = image.height-int(cy[0])
				term = cytomine_connection.get_term(ann.term[0])
				terms[term.name]=1
	key_t = terms.keys()
	
	job = cytomine_connection.update_job_status(job, status = job.RUNNING, status_comment = "Write in file", progress = 90)
	
	csv = open('%s%s.csv'%(parameters['cytomine_working_path'],str(parameters['cytomine_id_project'])),'w')
	
	csv.write('ID_IMAGE;')
	for i in range(len(key_t)):
		csv.write('%s_x;%s_y;'%(str(key_t[i]),str(key_t[i])))
	csv.write('\n')

	for image in images:
		csv.write('%s;'%str(image.id))
		for i in range(len(key_t)):
			if((key_t[i],image.id) in xpos):
				csv.write('%3.3f;%3.3f;'%(xpos[(key_t[i],image.id)],ypos[(key_t[i],image.id)]))
			else:
				csv.write('-1;-1;')
		csv.write('\n')

	csv.close()

	job_data = cytomine_connection.add_job_data(job, key=job.id, filename='%s%s.csv'%(parameters['cytomine_working_path'],str(parameters['cytomine_id_project'])))
	cytomine_connection.upload_job_data_file(job_data,'%s%s.csv'%(parameters['cytomine_working_path'],str(parameters['cytomine_id_project'])))

	job = cytomine_connection.update_job_status(job, status = job.TERMINATED, status_comment = "File available", progress = 100)
	print "File available"
	
if __name__ == '__main__':
	main(sys.argv[1:])
