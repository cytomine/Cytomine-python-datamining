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


__author__          = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__    = ["Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"


import cytomine
import os, optparse
from cytomine.models import *

import cPickle as pickle
import numpy as np

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from pyxit import pyxitstandalone
from pyxit.data import build_from_dir
from pyxit.estimator import PyxitClassifier, MAX_INT, _get_output_from_mask

import argparse, optparse
import time
from time import localtime, strftime

#Usage:
#This file download (dump) existing annotations from the server at specified dump_zoom
#It builds a segmentation model (using randomly extracted subwindows and extra-trees with multiple outputs) that tries to 
# discriminate between the predicted_terms (regrouped into one class),and all other terms (regrouped in a second class), 
# but without using terms specified in excluded_terms.
#You need to specify the Cytomine id_project, and the software id (as produced by the add_software.py script)
#See test-train.sh


#Cytomine parameters are given in command-line
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/home/maree/tmp/cytomine/annotations/', #local path to dump annotations,...
'cytomine_id_software' : 816476, #to be adapted
'cytomine_id_project' : 669418, #to be adapted
'cytomine_zoom_level' : 3, #zoom level: 0=maximum magnification
'cytomine_annotation_projects' : [669418],  #to be adapted: id of projets from which we dump annotations for learning
'cytomine_predict_terms' : [20202,4746,2171300,2171395], #to be adapted: terms regrouped into positive class
'cytomine_excluded_terms' : [5735, 4760, 28859,425818], #to be adapted: terms not used (e.g. section)
'cytomine_reviewed': True #to be adapted: do we retrieve reviewed annotations or regular user annotations
}

#pyxit parameters are described in the file pyxit/pyxitstandalone.py, see also test-train.sh
pyxit_parameters = {
'dir_ls' : "/",
'dir_ts' : "/",
'forest_shared_mem' : True,
#processing
'pyxit_n_jobs' : 10, #threads
#subwindows extraction
'pyxit_n_subwindows' : 100, #number of subwindows extracted per annotation
'pyxit_target_width' : 24,  #fixed size in segmentation mode
'pyxit_target_height' : 24, #fixed size in segmentation mode
'pyxit_interpolation' : 1, #interpolation used if subwindows are resized
'pyxit_transpose' : 1, #do we apply rotation/mirroring to subwindows (to enrich training set)
'pyxit_colorspace' : 2, # which colorspace do we use ?
'pyxit_fixed_size' : True, #fixed size in segmentation mode
'pyxit_save_to' : '',
#classifier parameters
'forest_n_estimators' : 10, #number of trees
'forest_max_features' : 28, #number of attributes considered at each node
'forest_min_samples_split' : 1, #nmin
'svm' : 0, #no svm in segmentation mode
'svm_c': 1.0,
}


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

def main(argv):
    # Define command line options
    p = optparse.OptionParser(description='Pyxit/Cytomine Segmentation Model Builder',
                              prog='PyXit Segmentation Model Builder (PYthon piXiT)')

    p.add_option("--cytomine_host", type="string", default = '', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
    p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
    p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
    p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
    p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
    p.add_option('--cytomine_annotation_projects', type="string", dest="cytomine_annotation_projects", help="Projects from which annotations are extracted")	
    p.add_option('--cytomine_predict_terms', type='string', default='0', dest='cytomine_predict_terms', help="term ids of predicted terms (=positive class in binary mode)")
    p.add_option('--cytomine_excluded_terms', type='string', default='0', dest='cytomine_excluded_terms', help="term ids of excluded terms")
    p.add_option('--cytomine_reviewed', type='string', default="False", dest="cytomine_reviewed", help="Get reviewed annotations only")
    p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
    p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
    p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit model directory") #future: get it from server db
    p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding") #future: get it from server db
    p.add_option('--pyxit_n_jobs', type='int', dest='pyxit_n_jobs', help="pyxit number of jobs for trees") #future: get it from server db
    p.add_option('--pyxit_n_subwindows', default=10, type="int", dest="pyxit_n_subwindows", help="number of subwindows")
    p.add_option('--pyxit_interpolation', default=2, type="int", dest="pyxit_interpolation", help="interpolation method 1,2,3,4")
    p.add_option('--pyxit_transpose', type="string", default="False", dest="pyxit_transpose", help="transpose subwindows")
    p.add_option('--pyxit_fixed_size', type="string", default="False", dest="pyxit_fixed_size", help="extract fixed size subwindows")
    p.add_option('--forest_n_estimators', default=10, type="int", dest="forest_n_estimators", help="number of base estimators (T)")
    p.add_option('--forest_max_features' , default=1, type="int", dest="forest_max_features", help="max features at test node (k)")
    p.add_option('--forest_min_samples_split', default=1, type="int", dest="forest_min_samples_split", help="minimum node sample size (nmin)")
    p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

    options, arguments = p.parse_args( args = argv)

    parameters['cytomine_host'] = options.cytomine_host	
    parameters['cytomine_public_key'] = options.cytomine_public_key
    parameters['cytomine_private_key'] = options.cytomine_private_key
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_working_path'] = options.cytomine_working_path	
    parameters['cytomine_base_path'] = options.cytomine_base_path
    parameters['cytomine_id_project'] = options.cytomine_id_project
    parameters['cytomine_id_software'] = options.cytomine_id_software
    parameters['cytomine_annotation_projects'] = map(int,options.cytomine_annotation_projects.split(','))
    parameters['cytomine_predict_terms'] = map(int,options.cytomine_predict_terms.split(','))
    parameters['cytomine_excluded_terms'] = map(int,options.cytomine_excluded_terms.split(','))
    parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
    parameters['cytomine_reviewed'] = str2bool(options.cytomine_reviewed)

   
    pyxit_parameters['pyxit_target_width'] = options.pyxit_target_width
    pyxit_parameters['pyxit_target_height'] = options.pyxit_target_height
    pyxit_parameters['pyxit_n_subwindows'] = options.pyxit_n_subwindows
    pyxit_parameters['pyxit_colorspace'] = options.pyxit_colorspace
    pyxit_parameters['pyxit_interpolation'] = options.pyxit_interpolation
    pyxit_parameters['pyxit_transpose'] = str2bool(options.pyxit_transpose)
    pyxit_parameters['pyxit_fixed_size'] = str2bool(options.pyxit_fixed_size)
    pyxit_parameters['forest_n_estimators'] = options.forest_n_estimators
    pyxit_parameters['forest_max_features'] = options.forest_max_features
    pyxit_parameters['forest_min_samples_split'] = options.forest_min_samples_split
    pyxit_parameters['pyxit_save_to'] = options.pyxit_save_to
    pyxit_parameters['pyxit_n_jobs'] = options.pyxit_n_jobs


    
    # Check for errors in the options
    if options.verbose:
      print "[pyxit.main] Options = ", options
    
    # Create JOB/USER/JOB
    conn = cytomine.Cytomine(parameters["cytomine_host"], 
                             parameters["cytomine_public_key"], 
                             parameters["cytomine_private_key"] , 
                             base_path = parameters['cytomine_base_path'], 
                             working_path = parameters['cytomine_working_path'], 
                             verbose= str2bool(options.verbose))

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


    pyxit_parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"], str(parameters['cytomine_annotation_projects']).replace(',','-').replace('[','').replace(']','').replace(' ',''), "zoom_level", str(parameters['cytomine_zoom_level']))
    if not os.path.exists(pyxit_parameters['dir_ls']):
        print "Creating annotation directory: %s" %pyxit_parameters['dir_ls']
        os.makedirs(pyxit_parameters['dir_ls'])
    time.sleep(2)
    job = conn.update_job_status(job, status_comment = "Publish software parameters values")
    all_params=pyxit_parameters
    all_params.update(parameters)
    job_parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), all_params)

    #Get annotation data
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Fetching data", progress = 0)    
    #Retrieve annotations from each annotation projects, either reviewed or unreviewed annotations
    annotations = None
    for prj in parameters['cytomine_annotation_projects']:
        if parameters["cytomine_reviewed"]:
            print "Retrieving reviewed annotations..."
            annotations_prj = conn.get_annotations(id_project = prj, reviewed_only=True)    
            print "Reviewed annotations: %d" %len(annotations_prj.data())
        else :
            print "Retrieving (unreviewed) annotations..."
            annotations_prj = conn.get_annotations(id_project = prj)
            print "(Unreviewed) annotations: %d" %len(annotations_prj.data())
        if not annotations :
            annotations = annotations_prj
        else : 
            annotations.data().extend(annotations_prj.data())
        print "Nb annotations so far... = %d" %len(annotations.data())
        time.sleep(3)
    print "Total annotations projects %s = %d" %(parameters['cytomine_annotation_projects'],len(annotations.data()))
    time.sleep(3)
    print "Predict terms / excluded terms"
    print parameters['cytomine_predict_terms']
    print parameters['cytomine_excluded_terms']
    time.sleep(3)
    annotations = conn.dump_annotations(annotations = annotations, get_image_url_func = Annotation.get_annotation_alpha_crop_url, dest_path = pyxit_parameters['dir_ls'], excluded_terms = parameters['cytomine_excluded_terms'], desired_zoom = parameters['cytomine_zoom_level'])



    #Build matrix (subwindows described by pixel values and output) for training
    project = conn.get_project(parameters['cytomine_id_project'])
    terms = conn.get_terms(project.ontology)
    map_classes = {} # build X, Y. Change initial problem into binary problem : "predict_terms" vs others
    for term in terms.data():
		if term.id in parameters['cytomine_predict_terms']:
			map_classes[term.id] = 1
		else :
			map_classes[term.id] = 0
    print pyxit_parameters
    
    #Prepare image matrix
    X, y = build_from_dir(pyxit_parameters['dir_ls'], map_classes)
    print "X length: %d " %len(X)
    print "Y length: %d " %len(y)
    time.sleep(5)
    #classes = np.unique(y)
    classes = [0,1]
    n_classes = len(classes)
    y_original = y
    y = np.searchsorted(classes, y)		


    # Instantiate classifiers
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "[pyxit.main] Initializing PyxitClassifier...", progress = 25)                    
    forest = ExtraTreesClassifier(n_estimators=pyxit_parameters['forest_n_estimators'],
                                  max_features=pyxit_parameters['forest_max_features'],
                                  min_samples_split=pyxit_parameters['forest_min_samples_split'],
                                  n_jobs=pyxit_parameters['pyxit_n_jobs'],
                                  verbose=True)

    pyxit = PyxitClassifier(base_estimator=forest,
                            n_subwindows=pyxit_parameters['pyxit_n_subwindows'],
                            min_size=0.0,#segmentation use fixed-size subwindows
                            max_size=1.0,#segmentation use fixed-size subwindows
                            target_width=pyxit_parameters['pyxit_target_width'],
                            target_height=pyxit_parameters['pyxit_target_height'],
                            interpolation=pyxit_parameters['pyxit_interpolation'],
                            transpose=pyxit_parameters['pyxit_transpose'],
                            colorspace=pyxit_parameters['pyxit_colorspace'],
                            fixed_size=pyxit_parameters['pyxit_fixed_size'],
                            n_jobs=pyxit_parameters['pyxit_n_jobs'],
                            verbose=True, 
                            get_output = _get_output_from_mask)
	
    
    if pyxit_parameters['pyxit_save_to']:
        d = os.path.dirname(pyxit_parameters['pyxit_save_to'])
        if not os.path.exists(d):
            os.makedirs(d)
        fd = open(pyxit_parameters['pyxit_save_to'], "wb")
        pickle.dump(classes, fd, protocol=pickle.HIGHEST_PROTOCOL)


    job = conn.update_job_status(job, status_comment = "[pyxit.main] Extracting %d subwindows from each image in %s" %(pyxit_parameters['pyxit_n_subwindows'],pyxit_parameters['dir_ls']), progress = 50)                        
    time.sleep(3)
    #Extract random subwindows in dumped annotations
    _X, _y = pyxit.extract_subwindows(X, y)    


    #Build pixel classifier
    job = conn.update_job_status(job, status_comment = "[pyxit.main] Fitting Pyxit Segmentation Model on %s", progress = 75)
    print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
    start = time.time()
    pyxit.fit(X, y, _X=_X, _y=_y)
    end = time.time()
    print "Elapsed time FIT: %d s" %(end-start)
    print "TIME : %s" %strftime("%Y-%m-%d %H:%M:%S", localtime())
    
    print "pyxit.base_estimator.n_classes_"
    print pyxit.base_estimator.n_classes_
    print "pyxit.base_estimator.classes_"
    print pyxit.base_estimator.classes_

    if options.verbose:
        print "----------------------------------------------------------------"
        print "[pyxit.main] Saving Pyxit Segmentation Model locally into %s" % pyxit_parameters['pyxit_save_to']
        print "----------------------------------------------------------------"

    #Save model on local disk
    if pyxit_parameters['pyxit_save_to']:
        pickle.dump(pyxit, fd, protocol=pickle.HIGHEST_PROTOCOL)

    if pyxit_parameters['pyxit_save_to']:
        fd.close()
	
    print "Not Publishing model in db.."
    #job_data = conn.add_job_data(job, "model", pyxit_parameters['pyxit_save_to'])
    
    job = conn.update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)    

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
