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
__contributors__    = ["Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"


#See test-predict.sh

import cytomine
import os, optparse
from cytomine.models import *
import cPickle as pickle
import numpy as np
import time
from pyxit import pyxitstandalone
from cytomine import models
from cytomine.utils import parameters_values_to_argv
import cv2

#Parameter values are set through command-line, see test-train.sh
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/bigdata/maree/cytomine/',
'cytomine_id_software' : 1,
'cytomine_id_project' : 1,
'cytomine_id_image' : 1,
'cytomine_id_userjob' : 1,
'cytomine_zoom_level' : 1,
'cytomine_dump_type' : 1,
}


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


def main(argv):
    # Define command line options
    p = optparse.OptionParser(description='Pyxit/Cytomine Classification Model Prediction',
                              prog='PyXit Classification Model Prediction (PYthon piXiT)')

    p.add_option("--cytomine_host", type="string", default = '', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
    p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
    p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
    p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit model file") #future: get it from server db
    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
    p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
    p.add_option('-i', '--cytomine_id_image', type="int", dest="cytomine_id_image", help="The Cytomine image identifier")	
    p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
    p.add_option('--cytomine_dump_type', type='int', dest='cytomine_dump_type', help="dump type of annotations (with/out alpha channel)")
    p.add_option('--cytomine_id_userjob', type="int", dest="cytomine_id_userjob", help="The Cytomine user (job) id of annotations to classify with the model")
    p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

    options, arguments = p.parse_args( args = argv)

    parameters['cytomine_host'] = options.cytomine_host	
    parameters['cytomine_public_key'] = options.cytomine_public_key
    parameters['cytomine_private_key'] = options.cytomine_private_key
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_working_path'] = options.cytomine_working_path	
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_id_software'] = options.cytomine_id_software
    #to define which annotations we will classify
    parameters['cytomine_id_project'] = options.cytomine_id_project
    parameters['cytomine_id_image'] = options.cytomine_id_image
    parameters['cytomine_id_userjob'] = options.cytomine_id_userjob
    #to define which resolution and image type we will classify
    parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
    parameters['cytomine_dump_type'] = options.cytomine_dump_type
    parameters['pyxit_save_to'] = options.pyxit_save_to


    
    print "[pyxit.main] Options = ", options
    
    # Create JOB/USER/JOB
    conn = cytomine.Cytomine(parameters["cytomine_host"], 
                             parameters["cytomine_public_key"], 
                             parameters["cytomine_private_key"] , 
                             base_path = parameters['cytomine_base_path'], 
                             working_path = parameters['cytomine_working_path'], 
                             verbose=str2bool(options.verbose))

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

    print "Fetching data..."
    job = conn.update_job_status(job, status_comment = "Publish software parameters values")
    if run_by_user_job==False:
        job_parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), parameters)
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Run...", progress = 0)

    #Image dump type (for classification use 1)
    if parameters['cytomine_dump_type']==1:
        annotation_get_func = Annotation.get_annotation_crop_url
    elif parameters['cytomine_dump_type']==2:
        annotation_get_func = Annotation.get_annotation_alpha_crop_url
    else:
        print "default annotation type crop"
        annotation_get_func = Annotation.get_annotation_crop_url  

    
        
    #Get description of annotations to predict (e.g. geometries created by a another object finder (e.g. threshold) job)
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Run (getting annotations)...", progress = 25)
    candidate_annotations = conn.get_annotations(id_user = parameters['cytomine_id_userjob'],
                                                 id_image = parameters['cytomine_id_image'], 
                                                 id_project = parameters['cytomine_id_project'],
                                                 showWKT=True, showMeta=True)

    print "Number of annotations to predict: %d" %len(candidate_annotations.data())
    time.sleep(2)

    #Create temporary dir to download annotation crops
    folder_name = "%s/annotations/project-%d/crops-candidates-%d-%d/zoom-%d/" % (parameters["cytomine_working_path"],
                                                                                 parameters["cytomine_id_project"], 
                                                                                 parameters["cytomine_id_image"], 
                                                                                 job.userJob,  #current job
                                                                                 parameters['cytomine_zoom_level'])
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    #Load Classifier model
    classifier = open(parameters['pyxit_save_to'], "r")
    classes = pickle.load(classifier)
    pyxit = pickle.load(classifier)
    print "Model: %s" %pyxit
    time.sleep(2)

    print "Dumping annotation cropped images to classify to %s" %folder_name
    annotation_mapping = {}
    for i, annotation in enumerate(candidate_annotations.data()):
        url = annotation_get_func(annotation, desired_zoom = parameters['cytomine_zoom_level'])
        filename = folder_name + str(annotation.id) + ".png"
        annotation_mapping[annotation.id] = filename 
        conn.fetch_url_into_file(url, filename, False, True)
        np_image = cv2.imread(filename, -1)
        if parameters['cytomine_dump_type']==2 and np_image is not None:
            alpha = np.array(np_image[:,:,3])
            image = np.array(np_image[:,:,0:3])
            image[alpha == 0] = (255,255,255)  #to replace alpha by white
            cv2.imwrite(filename, image)
        

    print "Building subwindows from ", os.path.dirname(os.path.dirname(folder_name))
    #Extract subwindows from all candidates annotations
    X, y = pyxitstandalone.build_from_dir(os.path.dirname(os.path.dirname(folder_name)))
    
    #Apply pyxit classifier model to X (parameters are already reused from model pkl file)
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Run (applying Classifier)...", progress = 50)
    y_proba = pyxit.predict_proba(X)
    y_predict = classes.take(np.argmax(y_proba, axis=1), axis=0)
    y_rate = np.max(y_proba, axis=1)

    #Creating new annotations on Cytomine with predicted terms by current classifier model
    for k, annotation in enumerate(candidate_annotations.data()) :
        filename = annotation_mapping[annotation.id]
        j = np.where(X == filename)[0][0]
        print "Annotation filename %s id: %d class: %d proba: %d" %(filename,annotation.id,int(y_predict[j]),y_rate[j])
        new_annotation = conn.add_annotation(annotation.location, parameters["cytomine_id_image"])
        conn.add_annotation_term(new_annotation.id, int(y_predict[j]), int(y_predict[j]), y_rate[j], annotation_term_model = models.AlgoAnnotationTerm)

    job = conn.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Finish Job..")
    print "END."

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
