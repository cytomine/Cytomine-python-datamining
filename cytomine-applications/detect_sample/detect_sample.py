# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
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
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"


import numpy as np
import sys

from cytomine_utilities import *
from cytomine import cytomine, models


from cytomine_utilities.objectfinder import ObjectFinder
from cytomine_utilities.utils import Utils
from shapely.geometry.polygon import Polygon
from shapely.wkt import dumps

from time import strftime, gmtime,sleep
import cv
import cv2
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




def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


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
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
        p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
        p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
        p.add_option('--cytomine_max_image_size', type='int', dest='cytomine_max_image_size', help="max_image_size to apply detection")
        p.add_option('--cytomine_erode_iterations', type='int', dest='cytomine_erode_iterations', help="number of erosions")
        p.add_option('--cytomine_athreshold_constant', type='int', dest='cytomine_athreshold_constant', help="increase/decrease value of threshold")
        p.add_option('--cytomine_athreshold_blocksize', type='int', dest='cytomine_athreshold_blocksize', help="size of neighboordhood to compute threshold")
        p.add_option('--cytomine_dilate_iterations', type='int', dest='cytomine_dilate_iterations', help="number of dilatations")
	p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
        p.add_option('--cytomine_id_image', type="int", dest="cytomine_id_image", help="The Cytomine image identifier")	
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on or off verbose mode")

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
        parameters['cytomine_max_image_size'] = options.cytomine_max_image_size
        parameters['cytomine_erode_iterations'] = options.cytomine_erode_iterations
        parameters['cytomine_dilate_iterations'] = options.cytomine_dilate_iterations
        parameters['cytomine_athreshold_constant'] = options.cytomine_athreshold_constant
        parameters['cytomine_athreshold_blocksize'] = options.cytomine_athreshold_blocksize
        
        

	print parameters
        
	# Check for errors in the options
	if options.verbose:
		print "[pyxit.main] Options = ", options
	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        #Connection to Cytomine server
	print "Connection to Cytomine server"
        conn = cytomine.Cytomine(parameters["cytomine_host"], 
                                 parameters["cytomine_public_key"], 
                                 parameters["cytomine_private_key"] , 
                                 base_path = parameters['cytomine_base_path'], 
                                 working_path = parameters['cytomine_working_path'], 
                                 verbose=str2bool(options.verbose))
        #Initialize parameters
	id_software = parameters['cytomine_id_software']
        print "id software : %d" %id_software
	software = conn.get_software(id_software)
	idProject = parameters['cytomine_id_project']
	idTerm = ['cytomine_id_predicted_term'] 
	project = conn.get_project(idProject)
	ontology = conn.get_ontology(project.ontology)
	terms = conn.get_terms(project.ontology)
	idSoftware = parameters['cytomine_id_software'] 


        #Dump images from project
        print "---------------------------------- DUMP images from project %d -------------------------------------------------" %idProject
        images = conn.dump_project_images(id_project = idProject, dest_path = "images", override = True, max_size = parameters['cytomine_max_image_size'])

	#Process each image to detect sample
	i = 0
	geometries = []


        #Create a new userjob if connected as human user
        current_user = conn.get_current_user()
        if current_user.algo==False:
            print "adduserJob..."
            user_job = conn.add_user_job(id_software, idProject)
            print "set_credentials..."
            conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
            print "done"
        else:
            user_job = current_user
            print "Already running as userjob"
        job = conn.get_job(user_job.job)
        job = conn.update_job_status(job, status_comment = "Create software parameters values...")
        job_parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), parameters)        

	for obj in images:
		if True:
                        print "Processing Cytomine imageid %d jobid %d" %(obj.id,job.userJob)
			job = conn.update_job_status(job, status = job.RUNNING, progress = i/len(images)*100, status_comment = "Processing images...")
                        
                        #Resize image
			print "Resizing and processing image %s" % obj.filename
                        ori_image = cv2.imread(obj.filename,0)
                        ori_height, ori_width = ori_image.shape
                        print "Downloaded thumbnail image: %dx%d" %(ori_width, ori_height)
                        max_dim = max(ori_width,ori_height)
                        resize_ratio = max_dim / parameters['cytomine_max_image_size']
			if resize_ratio > 1:
                                image = cv2.resize(image, (0,0), fx=int(ori_width / resize_ratio), fy=int(ori_height / resize_ratio))
                        else:
                                resize_ratio=1 #we do not upscale images
                                image = ori_image

                        #Threshold image using adaptive thresholding
                        new_height=int(ori_height/resize_ratio)
                        new_width=int(ori_width/resize_ratio)
                        cv_im = image
                        print cv_im.shape
                        cv_im_thres = cv2.adaptiveThreshold(cv_im, 255, cv.CV_ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                            cv.CV_THRESH_BINARY, 
                                                            parameters['cytomine_athreshold_blocksize'], 
                                                            parameters['cytomine_athreshold_constant'])

                        kernel = np.ones((5,5),np.uint8)
                        cv_im_erode = cv2.erode(cv_im_thres, kernel, iterations = parameters['cytomine_erode_iterations'])  #3
			cv_im_dilate = cv2.dilate(cv_im_erode, kernel, iterations = parameters['cytomine_dilate_iterations']) #3
                        cv_im = cv_im_dilate

                        #Save binarized image
			#image = Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())
			#image_width, image_height = cv.GetSize(cv_im)
			#print "Image sizes: %d %d" %(image_width,image_height)
			#filename = obj.filename.replace(".jpg", "_clo.png")
			#image.save(filename, "PNG")

                        #Find components, get a CV_RETR_LIST with all components
                        #Extend the image with borders (padding) to find contours touching the borders
                        ext=10
                        extended_cv_im= cv2.copyMakeBorder(cv_im,ext,ext,ext,ext,cv2.BORDER_CONSTANT,value=[255,255,255])
                        components = ObjectFinder(extended_cv_im).find_components_list()


                        #Convert components to real coordinates in the whole slide
                        image_height, image_width = cv_im.shape #cv2.getSize(extended_cv_im)
                        
 			zoom_factor = obj.width / float(image_width)
                        print "obj.width: %d" %obj.width
                        print "image: width %d height %d" %(image_width,image_height)
                        print "Apply zoom_factor to geometries: %f" %zoom_factor
			converted_components = []
			for component in components:
				converted_component = []
				for point in component[0]:
					x = point[0] - ext
					y = point[1] - ext
					x_at_maximum_zoom = x * zoom_factor
					y_at_maximum_zoom =  obj.height - (y * zoom_factor)
					point = (int(x_at_maximum_zoom), int(y_at_maximum_zoom))
					converted_component.append(point)

				converted_components.append(converted_component)

                        #Search and delete the biggest component ( = the whole slide)
			min_area = 4000000
			max_area = 999999999999999999
			biggest_geom = None
			print converted_components
			for component in converted_components:
				if not(biggest_geom):
					biggest_geom = component
				elif Polygon(biggest_geom).area < Polygon(component).area:
					biggest_geom = component

                        #Skip the biggest component and filter based on min/max size
			locations = []
			for component in converted_components:
				p = Polygon(component)
				if component == biggest_geom:
					continue
				if min_area and max_area:                
					if p.area > min_area and p.area < max_area:
						locations.append(p.wkt) 
				else:
					locations.append(p.wkt) 
                
			print "Image : %s " % obj.filename
			print "Nb annotations found : %d" % len(locations)


                        #Convert to geometries and upload annotations (with term) to Cytomine-Core
			for geometry in locations:
				print "Uploading geometry %s" % geometry
				annotation = conn.add_annotation(geometry, obj.id)
				print annotation
				if annotation:
					conn.add_annotation_term(annotation.id, parameters['cytomine_predict_term'], parameters['cytomine_predict_term'], 1.0, annotation_term_model = models.AlgoAnnotationTerm)
			geometries = []
			

                        i += 1

        job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
        job = None	
        print "END."

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())






