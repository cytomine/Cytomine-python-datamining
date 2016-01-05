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


__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__    = ["Gilles Louppe <g.louppe@gmail.com>", "Stévens Benjamin <b.stevens@ulg.ac.be>", "Olivier Caubo"]
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"



#exemple usage, see test-predict.sh
#This is a whole workflow (code to be redesigned in a future release): It can work within ROIs (e.g. tissue section),
#apply a segmentation model (pixel classifier) tile per tile, detect connected components, perform union of detected geometries 
#in the whole image, apply post-processing based on min/max are, apply a final classifier on geometries, 
#and finally output statistics (counts/area).

try:
    import Image, ImageStat
except:
    from PIL import Image, ImageStat

import sys
import time
import pickle
import copy
from progressbar import *
import os, optparse
from time import localtime, strftime
import socket

from shapely.geometry.polygon import Polygon
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from shapely.wkt import loads
import shapely.wkt
from shapely.geometry import box
import numpy as np
import matplotlib.path
from matplotlib.path import Path
import scipy.ndimage
import cv
import cv2
import math

import cytomine
from pyxit.data import build_from_dir
from pyxit.estimator import _get_image_data, _partition_images
from cytomine import cytomine, models
from cytomine_utilities.wholeslide import WholeSlide
from cytomine_utilities.objectfinder import ObjectFinder
from cytomine_utilities.reader import Bounds, CytomineReader
from cytomine_utilities.utils import Utils
from cytomine.models import ImageInstanceCollection


#Parameter values are now set through command-line
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/home/maree/tmp/cytomine/',
'cytomine_id_software' : 0,
'cytomine_id_project' : 0,
'cytomine_id_image' : None,
'cytomine_zoom_level' : 0,
'cytomine_tile_size' : 512,
'cytomine_tile_min_stddev' : 0,
'cytomine_tile_max_mean' : 255,
'cytomine_predict_term' : 0,
'cytomine_union' : False,
'cytomine_postproc' : False,
'cytomine_count' : False,
'cytomine_min_size' : 0,
'cytomine_max_size' : 1000000000,
'cytomine_roi_term': None,
'cytomine_reviewed_roi': None,
'pyxit_target_width' : 24, 
'pyxit_target_height' : 24,
'pyxit_predict_step' : 1,
'pyxit_save_to' : '/home/maree/tmp/cytomine/models/test.pkl',
'pyxit_colorspace' : 2,
'pyxit_nb_jobs' : 10,
'nb_jobs' : 20,
'publish_annotations': True ,

}



#-----------------------------------------------------------------------------------------------------------
#Functions


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


#For parallel extraction of subwindows in current tile
def _parallel_crop_boxes (y_roi, x_roi, image_filename, half_width, half_height, pyxit_colorspace):
    try:
        import Image
    except:
        from PIL import Image

    _X = []
    boxes = np.empty((len(x_roi)*len(y_roi), 4),dtype=np.int)
    i = 0
    image = Image.open(image_filename)
    for y in y_roi:     
        for x in x_roi:         
            min_x = int(x - half_width)
            min_y = int(y - half_height)
            max_x = int(x + half_width)
            max_y = int(y + half_height)        
            boxes[i] = min_x, min_y, max_x, max_y
            sub_window = image.crop(boxes[i])
            sub_window_data = _get_image_data(sub_window, pyxit_colorspace)
            _X.append(sub_window_data)
            i += 1
    return boxes, _X

#For parallel construction of confidence map in current tile
def _parallel_confidence_map(pixels, _Y, offset, boxes, tile_width, tile_height, n_classes, pyxit_target_width, pyxit_target_height):
    votes_class = np.zeros((tile_width, tile_height, n_classes))

    for i in pixels:
        inc_x = i % pyxit_target_width
        inc_y = i / pyxit_target_height
    
        for box_index, probas in enumerate(_Y[i-offset]):
            px = boxes[box_index][0] + inc_x
            py = boxes[box_index][1] + inc_y
            votes_class[py, px, :] += probas

    return votes_class                                                                                                          


#To convert a polyogn into a list of components
def polygon_2_component(polygon):
    exterior = list(polygon.exterior.coords)
    interiors = []
    for interior in polygon.interiors:
        interiors.append(list(interior.coords))
    return (exterior, interiors)


#To convert a union of roi polygons into a rasterized mask
def rasterize_tile_roi_union(nx, ny, points, local_tile_component, roi_annotations_union, whole_slide, reader):
    tile_component = whole_slide.convert_to_real_coordinates(whole_slide, [local_tile_component], reader.window_position, reader.zoom)[0]
    tile_polygon = shapely.geometry.Polygon(tile_component[0], tile_component[1])
    tile_roi_union = tile_polygon.intersection(roi_annotations_union)
    
    tile_roi_union_components = []
    if (tile_roi_union.geom_type == "Polygon"):
        tile_roi_union_components.append(polygon_2_component(tile_roi_union))
    if (tile_roi_union.geom_type == "MultiPolygon"):
        for geom in tile_roi_union.geoms:
            tile_roi_union_components.append(polygon_2_component(geom))
            
    local_tile_roi_union_components = whole_slide.convert_to_local_coordinates(whole_slide, tile_roi_union_components, reader.window_position, reader.zoom)
    local_tile_roi_union_polygons = [shapely.geometry.Polygon(component[0], component[1]) for component in local_tile_roi_union_components]
    
    local_tile_roi_union_raster = np.zeros((ny, nx), dtype=np.bool)
    for polygon in local_tile_roi_union_polygons:
        vertices = np.concatenate([np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors])
        #grid = points_inside_poly(points, vertices) #deprecated > matplotlib 1.2
        path = Path(vertices)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))
        local_tile_roi_union_raster |= grid
        
    return local_tile_roi_union_raster


#To remove unvalid polygon patterns
def process_mask(mask):
    # remove down-left to up-right diagonal pattern
    structure1 = np.zeros((3,3))
    structure1[0,2] = 1
    structure1[1,1] = 1
    structure2 = np.zeros((3,3))
    structure2[0,1] = 1
    structure2[1,2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    
    mask = mask - pattern_mask
    
    # remove up-left to down-right diagonal pattern
    structure1 = np.zeros((3,3))
    structure1[0,0] = 1
    structure1[1,1] = 1
    structure2 = np.zeros((3,3))
    structure2[0,1] = 1
    structure2[1,0] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    
    mask = mask - pattern_mask  
    #TODO the question is :
    # Does removing the second pattern can recreate the first one ? If so, how to avoid it? (iterative way?)
    
    
    # remove up line
    structure1 = np.zeros((3,3))
    structure1[2,1] = 1
    structure1[1,1] = 1
    structure2 = np.zeros((3,3))
    structure2[1,0] = 1
    structure2[1,2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    mask = mask - pattern_mask
    
    # remove down line
    structure1 = np.zeros((3,3))
    structure1[0,1] = 1
    structure1[1,1] = 1
    structure2 = np.zeros((3,3))
    structure2[1,0] = 1
    structure2[1,2] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    mask = mask - pattern_mask
      
    # remove left line
    structure1 = np.zeros((3,3))
    structure1[1,1] = 1
    structure1[1,2] = 1
    structure2 = np.zeros((3,3))
    structure2[0,1] = 1
    structure2[2,1] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    mask = mask - pattern_mask
    
    # remove right line
    structure1 = np.zeros((3,3))
    structure1[1,1] = 1
    structure1[1,0] = 1
    structure2 = np.zeros((3,3))
    structure2[0,1] = 1
    structure2[2,1] = 1
    pattern_mask = scipy.ndimage.morphology.binary_hit_or_miss(mask, structure1 = structure1, structure2 = structure2).astype(np.uint8)
    pattern_mask[pattern_mask==1] = 255
    pattern_mask[pattern_mask==0] = 0
    mask = mask - pattern_mask
    
    return mask
#-----------------------------------------------------------------------------------------------------------



def main(argv):
        current_path = os.getcwd() +'/'+ os.path.dirname(__file__)
	# Define command line options
        print "Main function"
	p = optparse.OptionParser(description='Cytomine Segmentation prediction',
                              prog='Cytomine segmentation prediction',
                              version='0.1')

	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
        p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
        #p.add_option('--cytomine_union', action="store_true", default=False, dest="cytomine_union", help="Turn on union of geometries")
        p.add_option('--cytomine_union', type="string", default="0", dest="cytomine_union", help="Turn on union of geometries")
        #p.add_option('--cytomine_postproc', action="store_true", default=False, dest="cytomine_postproc", help="Turn on postprocessing")
        p.add_option('--cytomine_postproc', type="string", default="0", dest="cytomine_postproc", help="Turn on postprocessing")
        #p.add_option('--cytomine_count', action="store_true", default=False, dest="cytomine_count", help="Turn on object counting")
        p.add_option('--cytomine_count', type="string", default="0", dest="cytomine_count", help="Turn on object counting")
        
        p.add_option('--cytomine_min_size', type="int", default=0, dest="cytomine_min_size", help="minimum size (area) of annotations")	
        p.add_option('--cytomine_max_size', type="int", default=10000000000, dest="cytomine_max_size", help="maximum size (area) of annotations")	
        #p.add_option('--cytomine_mask_internal_holes', action="store_true", default=False, dest="cytomine_mask_internal_holes", help="Turn on precise hole finding")
        p.add_option('--cytomine_mask_internal_holes', type='string', default="0", dest="cytomine_mask_internal_holes", help="Turn on precise hole finding")
	
        p.add_option('-i', '--cytomine_id_image', type='int', dest='cytomine_id_image', help="image id from cytomine", metavar='IMAGE')
        p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
        p.add_option('-t', '--cytomine_tile_size', type='int', dest='cytomine_tile_size', help="sliding tile size")
        p.add_option('--cytomine_tile_min_stddev', type='int', default=5, dest='cytomine_tile_min_stddev', help="tile minimum standard deviation")
        p.add_option('--cytomine_tile_max_mean', type='int', default=250, dest='cytomine_tile_max_mean', help="tile maximum mean")
        p.add_option('--cytomine_union_min_length', type='int', default=5, dest='cytomine_union_min_length', help="union")
        p.add_option('--cytomine_union_bufferoverlap', type='int', default=5, dest='cytomine_union_bufferoverlap', help="union")
        p.add_option('--cytomine_union_area', type='int', default=5, dest='cytomine_union_area', help="union")
        p.add_option('--cytomine_union_min_point_for_simplify', type='int', default=5, dest='cytomine_union_min_point_for_simplify', help="union")
        p.add_option('--cytomine_union_min_point', type='int', default=5, dest='cytomine_union_min_point', help="union")
        p.add_option('--cytomine_union_max_point', type='int', default=5, dest='cytomine_union_max_point', help="union")
        p.add_option('--cytomine_union_nb_zones_width', type='int', default=5, dest='cytomine_union_nb_zones_width', help="union")
        p.add_option('--cytomine_union_nb_zones_height', type='int', default=5, dest='cytomine_union_nb_zones_height', help="union")
        p.add_option('-j', '--nb_jobs', type='int', dest='nb_jobs', help="number of parallel jobs")
        p.add_option('--startx', type='int', default=0, dest='cytomine_startx', help="start x position")
        p.add_option('--starty', type='int', default=0, dest='cytomine_starty', help="start y position")
        p.add_option('--endx', type='int', dest='cytomine_endx', help="end x position")
        p.add_option('--endy', type='int', dest='cytomine_endy', help="end y position")
        p.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
        p.add_option('--cytomine_roi_term', type='string', dest='cytomine_roi_term', help="term id of region of interest where to count)")
        #p.add_option('--cytomine_reviewed_roi', action="store_true", default=False, dest="cytomine_reviewed_roi", help="Use reviewed roi only")
        p.add_option('--cytomine_reviewed_roi', type='string', default="0", dest="cytomine_reviewed_roi", help="Use reviewed roi only")
        
        p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
        p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
        p.add_option('--cytomine_predict_step', type='int', dest='cytomine_predict_step', help="pyxit step between successive subwindows")
        p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit segmentation model file") #future: get it from server db
        #p.add_option('--pyxit_post_classification', action="store_true", default=False, dest="pyxit_post_classification", help="pyxit post classification of candidate annotations")
        p.add_option('--pyxit_post_classification', type="string", default="0", dest="pyxit_post_classification", help="pyxit post classification of candidate annotations")
        
        p.add_option('--pyxit_post_classification_save_to', type='string', dest='pyxit_post_classification_save_to', help="pyxit post classification model file") #future: get it from server db
        p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding") #future: get it from server db
        p.add_option('--pyxit_nb_jobs', type='int', dest='pyxit_nb_jobs', help="pyxit number of jobs for trees") #future: get it from server db
	
        #p.add_option('--verbose', action="store_true", default=False, dest="verbose", help="Turn on verbose mode")
        p.add_option('--verbose', type='string', default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

        
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
        parameters['model_id_job'] = 0
        if options.cytomine_roi_term: 
            parameters['cytomine_roi_term'] = map(int,options.cytomine_roi_term.split(','))
        parameters['cytomine_reviewed_roi'] = str2bool(options.cytomine_reviewed_roi)
        parameters['cytomine_union'] = str2bool(options.cytomine_union)
        parameters['cytomine_postproc'] = str2bool(options.cytomine_postproc)
        parameters['cytomine_mask_internal_holes'] = str2bool(options.cytomine_mask_internal_holes)
        parameters['cytomine_count'] = str2bool(options.cytomine_count)
        if options.cytomine_min_size: 
            parameters['cytomine_min_size'] = options.cytomine_min_size
        if options.cytomine_max_size:
            parameters['cytomine_max_size'] = options.cytomine_max_size
        parameters['cytomine_predict_step'] = options.cytomine_predict_step
        parameters['pyxit_save_to'] = options.pyxit_save_to
        parameters['pyxit_post_classification'] = str2bool(options.pyxit_post_classification)
        parameters['pyxit_post_classification_save_to'] = options.pyxit_post_classification_save_to
        parameters['pyxit_colorspace'] = options.pyxit_colorspace
        parameters['pyxit_nb_jobs'] = options.pyxit_nb_jobs
        parameters['cytomine_nb_jobs'] = options.pyxit_nb_jobs
        parameters['cytomine_id_image'] = options.cytomine_id_image
        parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
        parameters['cytomine_tile_size'] = options.cytomine_tile_size
        parameters['cytomine_tile_min_stddev'] = options.cytomine_tile_min_stddev
        parameters['cytomine_tile_max_mean'] = options.cytomine_tile_max_mean
        parameters['cytomine_union_min_length'] = options.cytomine_union_min_length
        parameters['cytomine_union_bufferoverlap'] = options.cytomine_union_bufferoverlap
        parameters['cytomine_union_area'] = options.cytomine_union_area
        parameters['cytomine_union_min_point_for_simplify'] = options.cytomine_union_min_point_for_simplify
        parameters['cytomine_union_min_point'] = options.cytomine_union_min_point
        parameters['cytomine_union_max_point'] = options.cytomine_union_max_point
        parameters['cytomine_union_nb_zones_width'] = options.cytomine_union_nb_zones_width
        parameters['cytomine_union_nb_zones_height'] = options.cytomine_union_nb_zones_height
        parameters['cytomine_startx'] = options.cytomine_startx
        parameters['cytomine_starty'] = options.cytomine_starty
        parameters['cytomine_endx'] = options.cytomine_endx
        parameters['cytomine_endy'] = options.cytomine_endy
        parameters['nb_jobs'] = options.nb_jobs
        parameters['pyxit_target_width'] = options.pyxit_target_width
        parameters['pyxit_target_height'] = options.pyxit_target_height

        print parameters


	# Check for errors in the options
	if options.verbose:
		print "[pyxit.main] Options = ", options

        #Initialization
        pyxit_target_width = parameters['pyxit_target_width']
        pyxit_target_height = parameters['pyxit_target_height']
        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
        print "Loading prediction model (local)"
        fp = open(parameters["pyxit_save_to"], "r")
        print fp
        classes = pickle.load(fp)
        pyxit = pickle.load(fp)
        pyxit.n_jobs = parameters['nb_jobs'] #multithread subwindows extraction in pyxit
        pyxit.base_estimator.n_jobs= parameters['pyxit_nb_jobs']  #multithread tree propagation 
        #Reading parameters
        zoom = parameters['cytomine_zoom_level'] #int(sys.argv[2]) if sys.argv[2] else int(0)
        predictionstep= parameters['cytomine_predict_step'] #int(sys.argv[3])
        id_image= parameters['cytomine_id_image'] #int(sys.argv[1])


        #Create local directory to dump tiles
        local_dir = "%s/slides/project-%d/tiles/" % (parameters['cytomine_working_path'], parameters["cytomine_id_project"])
        if not os.path.exists(local_dir):
            print "Creating tile directory: %s" %local_dir
            os.makedirs(local_dir)



        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
        print "Parameters: %s" %parameters
        print "Extraction: prediction step (X,Y displacement between two successive subwindows): %s" %predictionstep

        #Cytomine connection
        print "Connection to Cytomine server"
        conn = cytomine.Cytomine(parameters["cytomine_host"], 
                                 parameters["cytomine_public_key"], 
                                 parameters["cytomine_private_key"] , 
                                 base_path = parameters['cytomine_base_path'], 
                                 working_path = parameters['cytomine_working_path'], 
                                 verbose= True)


        print "Create Job and UserJob..."
        id_software = parameters['cytomine_id_software']
        #Create a new userjob if connected as human user
        current_user = conn.get_current_user()
        run_by_user_job = False
        if current_user.algo==False:
            print "adduserJob..."
            user_job = conn.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
            print "set_credentials..."
            conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
            print "done"
        else:
            user_job = current_user
            print "Already running as userjob"
            run_by_user_job = True
        job = conn.get_job(user_job.job)

        job = conn.update_job_status(job, status_comment = "Publish software parameters values")
        if run_by_user_job==False:
            job_parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), parameters)        
        job = conn.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Loading data...")



        #loop for images in the project id or a subset ? -> better a subset provided by user
        image_instances = ImageInstanceCollection()
        image_instances.project  =  parameters['cytomine_id_project']
        image_instances  =  conn.fetch(image_instances)
        images = image_instances.data()
        print "Nb images in project: %d" %len(images)
        progress=0
        progress_delta = 100/len(images)
        i=0

        #Go through all images
        for image in images:
            id_image=image.id
            progress_msg = "Analyzing image %s (%d / %d )..." %(id_image,i,len(images))
            job = conn.update_job_status(job, status = job.RUNNING, progress = progress, status_comment = progress_msg)

            #print "image id: %d width: %d height: %d resolution: %f magnification: %d filename: %s" %(image.id,image.width,image.height,image.resolution,image.magnification,image.filename)

            #Create local object to access the remote whole slide
            print "Creating connector to Slide Image from Cytomine server"
            image_instance = conn.get_image_instance(id_image, True)
            whole_slide = WholeSlide(image_instance)
            print "Whole slide: %d x %d pixels" %(whole_slide.width,whole_slide.height)
            print "Done"


            #endx and endy allow to stop image analysis at a given x,y position  (for debugging)
            if not parameters['cytomine_endx'] and not parameters['cytomine_endy']:
                print "End is not defined."
                endx = whole_slide.width
                endy = whole_slide.height
            else:
                endx = parameters['cytomine_endx']
                endy = parameters['cytomine_endy']

            # initialize variables and tools for roi
            nx = parameters['cytomine_tile_size']
            ny = parameters['cytomine_tile_size']
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            local_tile_component = ([(0,0), (0,ny), (nx,ny), (nx,0), (0,0)], [])
        

            # We can apply the segmentation model either in the whole slide (including background area), or only within multiple ROIs (of a given term)
            # For example ROI could be generated first using a thresholding step to detect the tissue
            # Here we build a polygon union containing all roi_annotations locations (user or reviewed annotations) to later match tile with roi masks
            if parameters['cytomine_roi_term'] or parameters['cytomine_reviewed_roi']:
                if parameters['cytomine_reviewed_roi']:
                    #retrieve reviewed annotations for roi
                    roi_annotations = conn.get_annotations(id_image = id_image, 
                                                           id_term = str(parameters['cytomine_roi_term']).replace('[','').replace(']','').replace(' ',''), 
                                                           id_project=parameters['cytomine_id_project'], 
                                                           reviewed_only = True)
                else:
                    #retrieve annotations with roi term
                    roi_annotations = conn.get_annotations(id_image = id_image, 
                                                           id_term = str(parameters['cytomine_roi_term']).replace('[','').replace(']','').replace(' ',''), 
                                                           id_project=parameters['cytomine_id_project'])

            
                time.sleep(1)    
                roi_annotations_locations = []
                for simplified_roi_annotation in roi_annotations.data():
                    roi_annotation = conn.get_annotation(simplified_roi_annotation.id)
                    #roi_area_um += roi_annotation.area
                    assert shapely.wkt.loads(roi_annotation.location).is_valid, "one roi_annotation.location is not valid"
                    roi_annotations_locations.append(shapely.wkt.loads(roi_annotation.location))
                    roi_annotations_union = roi_annotations_locations[0]
                for annot in roi_annotations_locations[1:]:
                    roi_annotations_union = roi_annotations_union.union(annot)
            else: #no ROI used
                #We build a rectangular roi_mask corresponding to the whole image filled with ones
                print "We will process all tiles (no roi provided)"
                roi_mask = np.ones((ny, nx), dtype=np.bool)




            #Initiate the reader object which browse the whole slide image with tiles of size tile_size
            print "Initiating the Slide reader"
            reader = CytomineReader(conn,
                                    whole_slide,
                                    window_position = Bounds(parameters['cytomine_startx'],
                                                             parameters['cytomine_starty'],
                                                             parameters['cytomine_tile_size'],
                                                             parameters['cytomine_tile_size']),
                                    zoom = zoom,
                                    overlap = parameters['pyxit_target_width']+1)
            #opencv object image corresponding to a tile
            cv_image = cv.CreateImageHeader((reader.window_position.width, reader.window_position.height), cv.IPL_DEPTH_8U, 1)
            wsi=0
            geometries = []
        
            print "Starting browsing the image using tiles"
            #posx,posy,poswidth,posheight = reader.window_position.x, reader.window_position.y, reader.window_position.width,reader.window_position.height
            while True:
            
                #Browse the whole slide image with catch exception
                read = False
                while (not read) :
                    print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                    try :
                        reader.read(async = False)
                        read = True
                    except socket.error :
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        print socket.error
                        time.sleep(1)
                        continue

                    except socket.timeout :
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        print socket.timeout
                        time.sleep(1)
                        continue


               

                #Get rasterized roi mask to match with this tile (if no ROI used, the roi_mask was built before and corresponds to the whole image).
                if parameters['cytomine_roi_term']:
                    roi_mask = rasterize_tile_roi_union(nx, ny, points, local_tile_component, roi_annotations_union, whole_slide, reader)

                if np.count_nonzero(roi_mask):
                    print "nonzero roi_mask"
                    image=reader.data
                
                
                    #Get statistics about the current tile
                    print "Computing tile statistics"
                    posx,posy,poswidth,posheight = reader.window_position.x, reader.window_position.y, reader.window_position.width,reader.window_position.height
                    print "Tile zoom: %d posx: %d posy: %d poswidth: %d posheight: %d" % (zoom, posx, posy, poswidth, posheight)
                    tilemean = ImageStat.Stat(image).mean
                    print "Tile mean pixel values: %d %d %d" % (tilemean[0],tilemean[1],tilemean[2])
                    tilevar = ImageStat.Stat(image).var
                    print "Tile variance pixel values: %d %d %d" % (tilevar[0],tilevar[1],tilevar[2])
                    tilestddev = ImageStat.Stat(image).stddev
                    print "Tile stddev pixel values: %d %d %d" % (tilestddev[0],tilestddev[1],tilestddev[2])   
                    extrema = ImageStat.Stat(image).extrema
                    print extrema
                    print "extrema: min R:%d G:%d B:%d" % (extrema[0][0],extrema[1][0],extrema[2][0])


                    #Criteria to determine if tile is empty, specific to this application
                    mindev=parameters['cytomine_tile_min_stddev']
                    maxmean=parameters['cytomine_tile_max_mean']
                    if (((tilestddev[0] < mindev) and (tilestddev[1] < mindev) and (tilestddev[2] < mindev)) or ((tilemean[0] > maxmean) and (tilemean[1] > maxmean) and (tilemean[2] > maxmean))):
                        print "Tile empty (filtered by min stddev or max mean)"
            
                    else:
                        #This tile is not empty, we process it
                        #Add current tile annotation on server just for progress visualization purpose, not working
                        #current_tile = box(pow(2, zoom)*posx,
                        #                   whole_slide.height-pow(2, zoom)*posy-pow(2, zoom)*parameters['cytomine_tile_size'],
                        #                   pow(2, zoom)*posx+pow(2, zoom)*parameters['cytomine_tile_size'],
                        #                   whole_slide.height-pow(2, zoom)*posy)
                        #current_tile_annotation = conn.add_annotation(current_tile.wkt, id_image)
                        
                    
                        #Save the tile image locally
                        image_filename = "%s/slides/project-%d/tiles/%d-zoom_%d-tile_%d_x%d_y%d_w%d_h%d.png" % (parameters['cytomine_working_path'], parameters["cytomine_id_project"], id_image, zoom, wsi, posx, posy, poswidth, posheight)
                        image.save(image_filename,"PNG")
                
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        print "Tile file: %s" %image_filename
                        print "Extraction of subwindows in tile %d" %wsi
                        width, height = image.size
                        #nb_iter is the number of subwindows we extract in the tile, if predictionstep is 1 we extract all existing subwindows
                        nb_iter = ((height - 2*pyxit_target_height) * (width - 2*pyxit_target_width)) / (predictionstep*predictionstep)

                        #pbar = ProgressBar(maxval=nb_iter).start()
                        print "%d subwindows to extract" % nb_iter
                        half_width = math.floor(pyxit_target_width/2)
                        half_height = math.floor(pyxit_target_width/2)
                        #Coordinates of extracted subwindows
                        y_roi = range(pyxit_target_height/2,height - pyxit_target_height/2,predictionstep)
                        x_roi = range(pyxit_target_width/2, width - pyxit_target_width/2,predictionstep)
                
                        n_jobs = parameters['nb_jobs']
                        n_jobs, _, starts = _partition_images(n_jobs, len(y_roi))


                        #Parallel extraction of subwindows in the current tile
                        all_data = Parallel(n_jobs=n_jobs)(
                            delayed(_parallel_crop_boxes)(
                                y_roi[starts[i]:starts[i + 1]],
                                x_roi,
                                image_filename,
                                half_width,
                                half_height,
                                parameters['pyxit_colorspace'])
                            for i in xrange(n_jobs))

                        # Reduce
                        boxes = np.vstack(boxe for boxe, _ in all_data)
                        _X = np.vstack([X for _, X in all_data])
                

                        print "Prediction of subwindows for tile %d " % wsi
                        #Propagate subwindow feature vectors (X) into trees and get probabilities
                        _Y = pyxit.base_estimator.predict_proba(_X)

                        #Warning: we get output vectors for all classes for pixel (0,0) for all samples, then pixel predictions
                        #for pixel (0,1) for all samples, ... We do not get predictions samples after samples, but outputs after
                        #outputs

                        #<optimized code
                        print "Parallel construction of confidence map in current tile"
                        pixels = range(pyxit_target_width * pyxit_target_height)
                        n_jobs = parameters['nb_jobs']
                        n_jobs, _, starts = _partition_images(n_jobs, len(pixels))
        
                        all_votes_class = Parallel(n_jobs=n_jobs)(
                            delayed(_parallel_confidence_map)(
                                pixels[starts[i]:starts[i + 1]],
                                _Y[starts[i]:starts[i + 1]],
                                starts[i],
                                boxes,
                                width,
                                height,
                                pyxit.base_estimator.n_classes_[0],
                                pyxit_target_width,
                                pyxit_target_height)
                            for i in xrange(n_jobs))

                        votes_class = all_votes_class[0]

                        for v in all_votes_class[1:]:
                            votes_class += v
                        #optimizedcode>
        

                        print "Delete borders"
                        #Delete predictions at borders
                        for i in xrange(0,width):
                            for j in xrange(0,pyxit_target_height/2):
                                votes_class[i,j,:] = [1,0]
                            for j in xrange(height - pyxit_target_height/2,height): 
                                votes_class[i,j,:] = [1,0]



                        for j in xrange(0, height):
                            for i in xrange(0, pyxit_target_width / 2):
                                votes_class[i,j,:] = [1,0]
                            for i in xrange(width - pyxit_target_width/2, width):
                                votes_class[i,j,:] = [1,0]

                        #pbar.finish()
                        votes = np.argmax(votes_class, axis=2) * 255

                        #only predict in roi region based on roi mask
                        votes[np.logical_not(roi_mask)] = 0

                        #process mask
                        votes = process_mask(votes)


                        #current time
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        #Save of confidence map locally
                        print "Creating output tile file locally" 
                        output=Image.fromarray(np.uint8(votes))
                        outputfilename = "%s/slides/project-%d/tiles/%d-zoom_%d-tile_%d_xxOUTPUT-%dx%d.png" % (parameters["cytomine_working_path"],parameters["cytomine_id_project"], id_image, zoom, wsi, pyxit_target_width, pyxit_target_height)
                        output.save(outputfilename,"PNG")
                        print "Tile OUTPUT file: %s" %outputfilename


                        #Convert and transfer annotations of current tile
                        print "Find components"
                        #current time
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())


                        if parameters['cytomine_mask_internal_holes']:
                            #opencv cv2 new object finder with internal holes:
                            votes = votes.astype(np.uint8)
                            components = ObjectFinder(np.uint8(votes)).find_components()
                            components = whole_slide.convert_to_real_coordinates(whole_slide, components, reader.window_position, reader.zoom)
                            geometries.extend(Utils().get_geometries(components))
                        else:
                            #opencv old object finder without all internal contours:
                            cv.SetData(cv_image, output.tobytes())
                            components = ObjectFinder_(cv_image).find_components()
                            components = whole_slide.convert_to_real_coordinates_(whole_slide, components, reader.window_position, reader.zoom)
                            geometries.extend(Utils_().get_geometries(components))

                    
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
        
                        print "Uploading annotations..."
                        print "Number of geometries: %d" % len(geometries)
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        start = time.time()
                        #print geometries
                        print "------------------------------------------------------------------"
                        for geometry in geometries:
                            print "Uploading geometry %s" % geometry
                            startsingle = time.time()
                            uploaded = False
                            while(not uploaded) :
                                try :
                                    annotation = conn.add_annotation(geometry, id_image)
                                    uploaded = True
                                except socket.timeout, socket.error :
                                    print "socket timeout/error add_annotation"
                                    time.sleep(1)
                                    continue
                            endsingle = time.time()
                            print "Elapsed time ADD SINGLE ANNOTATION: %d" %(endsingle-startsingle)


                            print annotation
                            if annotation:
                                startsingle = time.time()
                                termed = False
                                while(not termed):
                                    try :
                                        conn.add_annotation_term(annotation.id, parameters['cytomine_predict_term'], parameters['cytomine_predict_term'], 1.0, annotation_term_model = models.AlgoAnnotationTerm)
                                        termed=True
                                    except socket.timeout, socket.error:
                                        print "socket timeout/error add_annotation_term"
                                        time.sleep(1)
                                        continue
                                endsingle = time.time()
                                print "Elapsed time ADD SINGLE ANNOTATION TERM: %d" %(endsingle-startsingle)
                        print "------------------------------------------------------------------"
                        #current time
                        end = time.time()
                        print "Elapsed time ADD ALL ANNOTATIONS: %d" %(end-start)
                        print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                        geometries = []

                    #Delete current tile annotation (progress visualization)
                    #conn.delete_annotation(current_tile_annotation.id)


                    
                else:
                    print "This tile (%05d) is not included in any ROI, so we skip processing" %wsi

            

                wsi+=1
                #if wsi<1: reader.next()
                #else: break
                if (not reader.next()) or ((reader.window_position.x > endx) and (reader.window_position.y > endy)): break
                #end of browsing the whole slide
    




            #Postprocessing to remove small/large annotations according to min/max area
            if parameters['cytomine_postproc']:
                print "POST-PROCESSING BEFORE UNION..."
                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                retrieved=False
                while(not retrieved):
                    try:
                        annotations = conn.get_annotations(id_user = job.userJob, id_image = id_image, id_project = parameters['cytomine_id_project'],showGIS=True)
                        retrieved=True
                    except socket.timeout, socket.error:
                        print "socket timeout/error get_annotations"
                        time.sleep(1)
                        continue
                
                #remove/edit useless annotations
                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                start = time.time()
                for annotation in annotations.data():
                    if annotation.area == 0 :
                        conn.delete_annotation(annotation.id)
                    else:
                        if annotation.area < parameters['cytomine_min_size'] : 
                            conn.delete_annotation(annotation.id)
                        elif annotation.area > parameters['cytomine_max_size'] : 
                            conn.delete_annotation(annotation.id)
                        else : 
                            print "OK KEEP ANNOTATION %d" %annotation.id
                            #if parameters['cytomine_simplify']:
                            #   print "ANNOTATION SIMPLIFICATION"
                            #  new_annotation = conn.add_annotation(annotation.location, annotation.image, minPoint=100, maxPoint=500)
                            # if new_annotation:
                            #    conn.add_annotation_term(new_annotation.id, predict_term, predict_term, 1.0, annotation_term_model = models.AlgoAnnotationTerm)
                            #   conn.delete_annotation(annotation.id) #delete old annotation
                            #predict_term = parameters['cytomine_predict_term']

                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                end = time.time()
                print "Elapsed time POST-PROCESS ALL ANNOTATIONS: %d" %(end-start)

        

            #Segmentation model was applied on individual tiles. We need to merge geometries generated from each tile.
            #We use a groovy/JTS script that downloads annotation geometries and perform union locally to relieve the Cytomine server
            if parameters['cytomine_union']:
                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                print "Union of polygons for job %d and image %d, term: %d" %(job.userJob,id_image,parameters['cytomine_predict_term'])
                start = time.time()
                host = parameters["cytomine_host"].replace("http://" , "")
                unioncommand = "groovy -cp \"../../lib/jars/*\" ../../lib/union4.groovy http://%s %s %s %d %d %d %d %d %d %d %d %d %d" %(host,
                                                                                                                                         user_job.publicKey,user_job.privateKey,
                                                                                                                                         id_image,job.userJob,
                                                                                                                                         parameters['cytomine_predict_term'], #union_term,
                                                                                                                                         parameters['cytomine_union_min_length'], #union_minlength,
                                                                                                                                         parameters['cytomine_union_bufferoverlap'], #union_bufferoverlap,
                                                                                                                                         parameters['cytomine_union_min_point_for_simplify'], #union_minPointForSimplify,
                                                                                                                                         parameters['cytomine_union_min_point'], #union_minPoint,
                                                                                                                                         parameters['cytomine_union_max_point'], #union_maxPoint,
                                                                                                                                         parameters['cytomine_union_nb_zones_width'], #union_nbzonesWidth,
                                                                                                                                         parameters['cytomine_union_nb_zones_height']) #union_nbzonesHeight)
                os.chdir(current_path)
                print unioncommand
                os.system(unioncommand)
                #old version was using a cytomine core webservice for union 
                #conn.union_polygons(job.userJob, id_image, union_term, union_minlength, union_area, buffer_length=union_bufferoverlap)
                end = time.time()
                print "Elapsed time UNION: %d s" %(end-start)
                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())


            #Perform classification of detected geometries using a classification model (pkl)
            if parameters['pyxit_post_classification']:
                print "POSTCLASSIFICATION OF all candidates"
                print "Create POSTCLASSIFICATION Job and UserJob..."
                conn2 = cytomine.Cytomine(parameters["cytomine_host"], 
                                          parameters["cytomine_public_key"], 
                                          parameters["cytomine_private_key"] , 
                                          base_path = parameters['cytomine_base_path'], 
                                          working_path = parameters['cytomine_working_path'], 
                                          verbose= True)
                id_software = parameters['cytomine_id_software']
                #create a new userjob related to the classification model
                pc_user_job = conn2.add_user_job(id_software, image_instance.project)
                conn2.set_credentials(str(pc_user_job.publicKey), str(pc_user_job.privateKey))
                pc_job = conn2.get_job(pc_user_job.job)
                pc_job = conn2.update_job_status(pc_job, status_comment = "Create software parameters values...")
                pc_job = conn2.update_job_status(pc_job, status = pc_job.RUNNING, progress = 0, status_comment = "Loading data...")
                #Retrieve locally annotations from Cytomine core produced by the segmentation job as candidates
                candidate_annotations = conn2.get_annotations(id_user = job.userJob, id_image = id_image, id_term = parameters['cytomine_predict_term'], showWKT=True, showMeta=True)
                nb_candidate_annotations = len(candidate_annotations.data())
                folder_name = "%s/slides/project-%d/tiles/crops-candidates-%d-%d/zoom-%d/" % (parameters["cytomine_working_path"],parameters["cytomine_id_project"], id_image, job.userJob, 0)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                annotation_mapping = {}
                for i, annotation in enumerate(candidate_annotations.data()):
                    url = annotation.get_annotation_alpha_crop_url(parameters['cytomine_predict_term'], desired_zoom = 0)
                    filename = folder_name + str(annotation.id) + ".png"  #str(i)
                    annotation_mapping[annotation.id] = filename 
                    conn2.fetch_url_into_file(url, filename, False, True)
                    np_image = cv2.imread(filename, -1)
                    if np_image is not None :
                        alpha = np.array(np_image[:,:,3])
                        image = np.array(np_image[:,:,0:3])
                    #image[alpha == 0] = (255,255,255)  #to replace surrounding by white
                    cv2.imwrite(filename, image)
                print "Building attributes from ", os.path.dirname(os.path.dirname(folder_name))
                #Extract subwindows from all candidates
                X, y = build_from_dir(os.path.dirname(os.path.dirname(folder_name)))
                post_fp = open(parameters['pyxit_post_classification_save_to'], "r")
                classes = pickle.load(post_fp)
                pyxit = pickle.load(post_fp)
                print pyxit
                time.sleep(3)
                #pyxit parameters are in the model file
                y_proba = pyxit.predict_proba(X)
                y_predict = classes.take(np.argmax(y_proba, axis=1), axis=0)
                y_rate = np.max(y_proba, axis=1)
                #We classify each candidate annotation and keep only those predicted as cytomine_predict_term
                for k, annotation in enumerate(candidate_annotations.data()) :
                    filename = annotation_mapping[annotation.id]
                    j = np.where(X == filename)[0][0]
                    if int(y_predict[j])==parameters['cytomine_predict_term']:
                        print "POSTCLASSIFICATION Annotation KEPT id: %d class: %d proba: %d" %(annotation.id,int(y_predict[j]),y_rate[j])
                    else: 
                        print "POSTCLASSIFICATION Annotation REJECTED id: %d class: %d proba: %d" %(annotation.id,int(y_predict[j]),y_rate[j])
                    new_annotation = conn2.addAnnotation(annotation.location, id_image)
                    conn2.addAnnotationTerm(new_annotation.id, int(y_predict[j]), int(y_predict[j]), y_rate[j], annotation_term_model = models.AlgoAnnotationTerm)

                print "POSTCLASSIFICATION END."
                print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
                pc_job = conn.update_job_status(pc_job, status = pc_job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
                #...
                    
                
            #Perform stats (counting) in roi area
            if parameters['cytomine_count'] and parameters['cytomine_roi_term']:
                if parameters['pyxit_post_classification']:
                    id_job = pc_job.userJob
                else:
                    id_job = job.userJob
                print "COUNTING..."    
                #Count number of annotations in roi area
                #Get Rois
                roi_annotations = conn.get_annotations(id_image = id_image, 
                                                       id_term = str(parameters['cytomine_roi_term']).replace('[','').replace(']','').replace(' ',''), 
                                                       id_project=parameters['cytomine_id_project'])
                #Count included annotations (term = predict_term) in each ROI
                for roi_annotation in roi_annotations.data():
                    included_annotations = conn.included_annotations(id_image = id_image, id_user = id_job, id_terms = parameters['cytomine_predict_term'], id_annotation_roi = roi_annotation.id)
                    print "STATSImageID %d name %s: Number of annotations (term: %d) included in ROI %d: %d" %(id_image,image_instance.originalFilename,parameters['cytomine_predict_term'],roi_annotation.id,len(included_annotations.data()))
                    roi_annot_descr = conn.get_annotation(roi_annotation.id)
                    print "STATSImageID %d ROI area: %d" %(id_image,roi_annot_descr.area)

                
            print "END image %d." %i
            progress += progress_delta
            i+=1

        job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
        sys.exit()    


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
