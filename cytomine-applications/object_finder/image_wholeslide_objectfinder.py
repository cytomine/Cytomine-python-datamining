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


__author__          = "Marée Raphaël <raphael.maree@ulg.ac.be>" 
__contributors__    = ["Stévens Benjamin <b.stevens@ulg.ac.be>"]                
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"



import os, sys, time
import optparse
import cv
import pickle

from cytomine import Cytomine, models
from cytomine_utilities.filter import AdaptiveThresholdFilter, BinaryFilter, OtsuFilter
from cytomine_utilities.objectfinder import ObjectFinder
from cytomine_utilities.reader import Bounds, CytomineReader
from cytomine_utilities.utils import Utils
from cytomine_utilities.wholeslide import WholeSlide


#Parameter values are now set through command-line
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : 'XXX',
'cytomine_id_software' : 0,
'cytomine_id_project' : 0,
'cytomine_id_image' : None,
'cytomine_zoom_level' : 0,
'cytomine_tile_size' : 512,
'cytomine_predict_term' : 0,
}



def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")



class Enum(set):
    def __getattr__(self, name):
        if name in self:
            return name
        raise AttributeError

def main(argv):

    current_path = os.getcwd() +'/'+ os.path.dirname(__file__)

    #Available filters
    filters = { 'binary' : BinaryFilter(), 'adaptive' : AdaptiveThresholdFilter(), 'otsu' : OtsuFilter()}

    parser = optparse.OptionParser(description='Cytomine Datamining',
                              prog='cytomining',
                              version='cytomining 0.1')
    parser.add_option('-i', '--cytomine_id_image', type='int', dest='cytomine_id_image',
                  help='image id from cytomine', metavar='IMAGE')
    parser.add_option('--cytomine_host', dest='cytomine_host', help='cytomine_host')
    parser.add_option('--cytomine_public_key', dest='cytomine_public_key', help='cytomine_public_key')
    parser.add_option('--cytomine_private_key', dest='cytomine_private_key', help='cytomine_private_key')
    parser.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
    parser.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
    parser.add_option('--cytomine_predict_term', type='int', dest='cytomine_predict_term', help="term id of predicted term (binary mode)")
    parser.add_option('--cytomine_base_path', dest='cytomine_base_path', help='cytomine base path')
    parser.add_option('--cytomine_working_path', dest='cytomine_working_path', help='cytomine_working_path base path')

    parser.add_option('--cytomine_zoom_level', dest='cytomine_zoom_level', type = 'int', help='(auto mode only) Zoom. 0 value is maximum zoom')
    parser.add_option('--cytomine_filter', dest='cytomine_filter', type = 'choice', choices = filters.keys())
    parser.add_option('--cytomine_min_area', dest='cytomine_min_area', type = 'int', help='min area of detected object in pixel at maximum zoom')
    parser.add_option('--cytomine_max_area', dest='cytomine_max_area', type = 'int', help='max area of detected object in pixel at maximum zoom')
    parser.add_option('--cytomine_tile_size', dest='cytomine_tile_size', type = 'int', help='window_size (sliding window). default is 1024')
    parser.add_option('--cytomine_tile_overlap', dest='cytomine_tile_overlap', type = 'int', help='overlap between two sliding window position in pixels. default is 0')

    parser.add_option('--cytomine_union_min_length', type='int', default=5, dest='cytomine_union_min_length', help="union")
    parser.add_option('--cytomine_union_bufferoverlap', type='int', default=5, dest='cytomine_union_bufferoverlap', help="union")
    parser.add_option('--cytomine_union_area', type='int', default=5, dest='cytomine_union_area', help="union")
    parser.add_option('--cytomine_union_min_point_for_simplify', type='int', default=5, dest='cytomine_union_min_point_for_simplify', help="union")
    parser.add_option('--cytomine_union_min_point', type='int', default=5, dest='cytomine_union_min_point', help="union")
    parser.add_option('--cytomine_union_max_point', type='int', default=5, dest='cytomine_union_max_point', help="union")
    parser.add_option('--cytomine_union_nb_zones_width', type='int', default=5, dest='cytomine_union_nb_zones_width', help="union")
    parser.add_option('--cytomine_union_nb_zones_height', type='int', default=5, dest='cytomine_union_nb_zones_height', help="union")
    parser.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

    options, arguments = parser.parse_args( args = argv)


    #copy options
    parameters['cytomine_host'] = options.cytomine_host	
    parameters['cytomine_public_key'] = options.cytomine_public_key
    parameters['cytomine_private_key'] = options.cytomine_private_key
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_working_path'] = options.cytomine_working_path	
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_id_project'] = options.cytomine_id_project
    parameters['cytomine_id_image'] = options.cytomine_id_image
    parameters['cytomine_id_software'] = options.cytomine_id_software
    parameters['cytomine_predict_term'] = options.cytomine_predict_term if options.cytomine_predict_term else int(0)
    parameters['cytomine_union_min_length'] = options.cytomine_union_min_length
    parameters['cytomine_union_bufferoverlap'] = options.cytomine_union_bufferoverlap
    parameters['cytomine_union_area'] = options.cytomine_union_area
    parameters['cytomine_union_min_point_for_simplify'] = options.cytomine_union_min_point_for_simplify
    parameters['cytomine_union_min_point'] = options.cytomine_union_min_point
    parameters['cytomine_union_max_point'] = options.cytomine_union_max_point
    parameters['cytomine_union_nb_zones_width'] = options.cytomine_union_nb_zones_width
    parameters['cytomine_union_nb_zones_height'] = options.cytomine_union_nb_zones_height
    parameters['cytomine_zoom_level'] = options.cytomine_zoom_level if options.cytomine_zoom_level else int(0)
    parameters['cytomine_tile_size'] = options.cytomine_tile_size if options.cytomine_tile_size else 1024
    parameters['cytomine_min_area'] = options.cytomine_min_area if options.cytomine_min_area else (50*50 / (2**parameters['cytomine_zoom_level'])) #25x25 pixels at zoom 0
    parameters['cytomine_max_area'] = options.cytomine_min_area if options.cytomine_min_area else (100000*100000 / (2**parameters['cytomine_zoom_level'])) #100000x100000 pixels at zoom 0
    parameters['cytomine_filter'] = options.cytomine_filter
    parameters['cytomine_tile_overlap'] = options.cytomine_tile_overlap if options.cytomine_tile_overlap else int(0)    
    
    
    #Get filter object
    filter = filters.get(options.cytomine_filter)
    
    # Init Cytomine-Core connection and reader object
    print "Connection to Cytomine server"
    conn = Cytomine(parameters["cytomine_host"], 
                             parameters["cytomine_public_key"], 
                             parameters["cytomine_private_key"] , 
                             base_path = parameters['cytomine_base_path'], 
                             working_path = parameters['cytomine_working_path'], 
                             verbose= str2bool(options.verbose))
    whole_slide = WholeSlide(conn.get_image_instance(parameters['cytomine_id_image'], True))
    async = False #True is experimental
    reader = CytomineReader(conn, whole_slide, window_position = Bounds(0,0, parameters['cytomine_tile_size'], parameters['cytomine_tile_size']), zoom = parameters['cytomine_zoom_level'], overlap = parameters['cytomine_tile_overlap'])
    cv_image = cv.CreateImageHeader((reader.window_position.width, reader.window_position.height), cv.IPL_DEPTH_8U, 3)
    reader.window_position = Bounds(0, 0, reader.window_position.width, reader.window_position.height)

    #Create a new userjob if connected as human user
    current_user = conn.get_current_user()
    if current_user.algo==False:
        print "adduserJob..."
        user_job = conn.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
        print "set_credentials..."
        conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
        print "done"
    else:
        user_job = current_user
        print "Already running as userjob"
    job = conn.get_job(user_job.job)
    job = conn.update_job_status(job, status_comment = "Create software parameters values...")
    job_parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), parameters)       



    #Browse the slide using reader
    i = 0
    geometries = []
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Fetching data", progress = 0)
    while True:
        #Read next tile
        reader.read(async = async)
        image=reader.data
        #Saving tile image locally
        tile_filename = "%s/image-%d-zoom-%d-tile-%d-x-%d-y-%d.png" %(parameters['cytomine_working_path'],
                                                                      parameters['cytomine_id_image'],
                                                                      parameters['cytomine_zoom_level']
                                                                      ,i
                                                                      ,reader.window_position.x,
                                                                      reader.window_position.y)
        image.save(tile_filename,"PNG")
        #Apply filtering
        cv.SetData(cv_image, reader.result().tobytes())
        filtered_cv_image = filter.process(cv_image)
        i += 1
        #Detect connected components
        components = ObjectFinder(filtered_cv_image).find_components()
        #Convert local coordinates (from the tile image) to global coordinates (the whole slide)
        components = whole_slide.convert_to_real_coordinates(whole_slide, components, reader.window_position, reader.zoom)
        geometries.extend(Utils().get_geometries(components, parameters['cytomine_min_area'], parameters['cytomine_max_area']))
            
        
        #Upload annotations (geometries corresponding to connected components) to Cytomine core
        if parameters['cytomine_predict_term']>0:
            #Upload each geometry and add predict term
            for geometry in geometries:
                annotation = conn.add_annotation(geometry, parameters['cytomine_id_image'])
                if annotation:
                    conn.add_annotation_term(annotation.id, parameters['cytomine_predict_term'], parameters['cytomine_predict_term'], 1.0, annotation_term_model = models.AlgoAnnotationTerm)
        else:
        #Upload all geometries at once (without term)
            conn.add_annotations(geometries, parameters['cytomine_id_image'])
        
        geometries = []
        if not reader.next(): break
     

    host = parameters["cytomine_host"].replace("http://" , "")    
    #Union of geometries (because geometries are computed locally in each time but objects (e.g. cell clusters) might overlap several tiles)
    print "Union of polygons for job %d and image %d, term: %d" %(job.userJob,parameters['cytomine_id_image'],
                                                                  parameters['cytomine_predict_term'])
    unioncommand = "groovy -cp \"../../lib/jars/*\" ../../lib/union4.groovy http://%s %s %s %d %d %d %d %d %d %d %d %d %d" %(host,
                                                                                                                                   user_job.publicKey,user_job.privateKey,
                                                                                                                                   parameters['cytomine_id_image'],
                                                                                                                                   job.userJob,
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

    #save coordinates of detected geometries to local file
    #output = open(os.path.join(cytomine_working_path,save_to), 'wb')
    #pickle.dump(geometries, output, protocol=pickle.HIGHEST_PROTOCOL)
    #output.close()
    job = conn.update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)
    print "END"

if __name__ == '__main__':
    main(sys.argv[1:])
