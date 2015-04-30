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


__author__          = "Stévens Benjamin <b.stevens@ulg.ac.be>"
__contributors__    = ["Marée Raphael <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import numpy
import sys


from cytomine_utilities import *
from cytomine import cytomine, models

#from object_finder.image.utils import Utils
#from object_finder.image.reader import Bounds, CytomineReader
#from object_finder.image.objectfinder import ObjectFinder
#from object_finder.image.wholeslide import WholeSlide

from objectfinder_list import ObjectFinderList


from shapely.geometry.polygon import Polygon
from shapely.wkt import dumps

import Image
from time import strftime, gmtime,sleep
import cv
import os, optparse


parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/home/maree/tmp/cytomine/',
'cytomine_id_software' : 0,
'cytomine_id_project' : 0,
'cytomine_predict_term' : 0,
'cytomine_id_image' : 0,
}


#ISO DATA
def getThreshold(data):
    max_value = len(data) - 1
    result = -1

    _min = 0
    while (data[_min] == 0) and (_min < max_value):
        _min += 1

    _max = max_value
    while (data[_max] == 0) and (_max > 0):
        _max -= 1
        
    if _min >= _max:
        return len(data) / 2

    movingIndex = _min
    
    while True:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for i in range(_min, movingIndex + 1):
            sum1 += i * data[i]
            sum2 += data[i]
        for i in range(movingIndex + 1, _max):
            sum3 += i * data[i]
            sum4 += data[i]
        result = (sum1 / sum2 + sum3 / sum4) / 2.0
        movingIndex += 1
        if movingIndex + 1 > result or movingIndex >= _max-1:
            break
    
    return int(round(result))




def main(argv):
	# Define command line options
        print "Main function"
	p = optparse.OptionParser(description='Cytomine Detect Sample',
                              prog='Cytomine Detect Sample on Slide',
                              version='0.1')

	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = 'XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = 'YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
        p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
	p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
        p.add_option('--cytomine_id_image', type="int", dest="cytomine_id_image", help="The Cytomine image identifier")	
	p.add_option('--verbose', action="store_true", default=False, dest="verbose", help="Turn on verbose mode")

	options, arguments = p.parse_args( args = argv)

	parameters['cytomine_host'] = options.cytomine_host	
        parameters['cytomine_public_key'] = options.cytomine_public_key
        parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_base_path'] = options.cytomine_base_path
        parameters['cytomine_working_path'] = options.cytomine_working_path	
        parameters['cytomine_base_path'] = options.cytomine_base_path
        parameters['cytomine_id_project'] = options.cytomine_id_project
        parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_predict_term'] = options.cytomine_predict_term
        parameters['cytomine_id_image'] = options.cytomine_id_image

	print parameters
        
	# Check for errors in the options
	if options.verbose:
		print "[pyxit.main] Options = ", options

	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
	print "Connection to Cytomine server"
        conn = cytomine.Cytomine(parameters["cytomine_host"], 
                                 parameters["cytomine_public_key"], 
                                 parameters["cytomine_private_key"] , 
                                 base_path = parameters['cytomine_base_path'], 
                                 working_path = parameters['cytomine_working_path'], 
                                 verbose= True)
        #Initialize parameters
	id_software = parameters['cytomine_id_software']
        print "id software : %d" %id_software
	software = conn.getSoftware(id_software)
	idProject = parameters['cytomine_id_project']
	idTerm = ['cytomine_id_predicted_term'] #5735
	project = conn.getProject(idProject)
	ontology = conn.getOntology(project.ontology)
	terms = conn.getTerms(project.ontology)
	idSoftware = parameters['cytomine_id_software'] #321655
	#idUser = parameters['cytomine_id_user'] #18

        #Dump images from project
        print "---------------------------------- DUMP image from project %d -------------------------------------------------" %idProject
	images = conn.dump_project_images(id_project = idProject, dest_path = "images", override = True, max_size = 2048)

	#########MAIN######## #
	i = 0
	geometries = []

#        print "Desired image id: %d" %['cytomine_id_image']
        #print "Image id : %d" %obj.id
        
        
	for obj in images:
                #if (obj.id == parameters["cytomine_id_image"] or parameters["cytomine_id_image"]==0):
		if True:


			print "Processing image %d " % obj.id

                        conn = cytomine.Cytomine(parameters["cytomine_host"], 
                                 parameters["cytomine_public_key"], 
                                 parameters["cytomine_private_key"] , 
                                 base_path = parameters['cytomine_base_path'], 
                                 working_path = parameters['cytomine_working_path'], 
                                 verbose= True)


                        #Create userJob for this image
			image_instance = conn.getImageInstance(obj.id)
			print "adduserJob..."
			user_job = conn.addUserJob(id_software, image_instance.project)
			print "done..."
			print "set_credentials..."
			conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
			print "done"
			job = conn.getJob(user_job.job)
			job = conn.update_job_status(job, status_comment = "Create software parameters values...")
			job = conn.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Loading data...")


                        #Resize image
			print "image %s" % obj.filename
			sleep(2)
			ori_image = Image.open(obj.filename)
			image = ori_image.convert('L')
			width_ratio = image.size[0] / 2000   #chimio:500 #default: 2000 #lung lido:1000  lung: 2000
			if width_ratio > 1:
				image = image.resize((int(image.size[0] / width_ratio), int(image.size[1] / width_ratio)))

			cv_ori = cv.CreateImageHeader(image.size, cv.IPL_DEPTH_8U, 3)
			cv.SetData(cv_ori, image.tostring())

                        #Threshold image using image global statistics
			cv_im = cv.CreateImageHeader(image.size, cv.IPL_DEPTH_8U, 1)
			cv.SetData(cv_im, image.tostring())
                        #add 10 to threshold will take more around the sample, while substracting will take less than the sample
			threshold = getThreshold(image.histogram()) +35 #+10  #chimio: -10 #+ 10  # +35 lung
			print "Threshold: %d" % threshold
			cv.Threshold(cv_im, cv_im, threshold, 255, cv.CV_THRESH_BINARY)
                        #cv.AdaptiveThreshold(cv_im, cv_im, 255, cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv.CV_THRESH_BINARY, 51, 5)
                        #ellipse = cv.CreateStructuringElementEx(15,15,0,0, cv.CV_SHAPE_ELLIPSE)
			
                        #default: erode 3, dilate 3; chimio: 3 4
                        cv.Erode(cv_im, cv_im, iterations = 3, element = None)
			cv.Dilate(cv_im, cv_im, iterations = 3, element = None)
        

                        #Save binarized image
			image = Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())
			image_width, image_height = cv.GetSize(cv_im)
			print "Image sizes: %d %d" %(image_width,image_height)
			filename = obj.filename.replace(".jpg", "_clo.png")
			image.save(filename, "PNG")


                        #Find components, get a CV_RETR_LIST with all components
			components = ObjectFinderList(cv_im).find_components()

                        #Convert component to real coordinates
			zoom_factor = (obj.width / image_width) 			
			converted_components = []
			for component in components:
				converted_component = []
				for point in component:
					x = point[0]
					y = point[1]
                
					x_at_maximum_zoom = x * zoom_factor
					y_at_maximum_zoom =  obj.height - (y * zoom_factor)
					point = (int(x_at_maximum_zoom), int(y_at_maximum_zoom))
					converted_component.append(point)

				converted_components.append(converted_component)

                        #Look for the biggest component ( = the whole slide)
			min_area = 4000000
			max_area = 999999999999999999
			biggest_geom = None
			print converted_components
			for component in converted_components:
				if not(biggest_geom):
					biggest_geom = component
				elif Polygon(biggest_geom).area < Polygon(component).area:
					biggest_geom = component

                        #Skip the biggest component
			locations = []
			for component in converted_components:
				p = Polygon(component)
				if component == biggest_geom:
					continue
				if min_area and max_area:                
					if p.area > min_area and p.area < max_area:
						locations.append(component)
				else:
					locations.append(component)
                
			print "Image : %s " % obj.filename
			print "Nb annotations : %d" % len(locations)


                        #Convert to geometries and upload annotations (with term)
			geometries.extend(Utils_().get_geometries(locations, min_area, max_area))
			for geometry in geometries:
				print "Uploading geometry %s" % geometry
				annotation = conn.addAnnotation(geometry, obj.id)
				print annotation
				if annotation:
					conn.addAnnotationTerm(annotation.id, parameters['cytomine_predict_term'], parameters['cytomine_predict_term'], 1.0, annotation_term_model = models.AlgoAnnotationTerm)
			geometries = []
			job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
			job = None	
                        #sys.exit()

		i += 1




if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())






