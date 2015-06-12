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
import time

from pyxit import pyxitstandalone
from cytomine.utils import parameters_values_to_argv


#Parameter values are set through command-line, see test-train.sh
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '/home/maree/tmp/cytomine/annotations/',
'cytomine_id_software' : 1,
'cytomine_id_project' : 1,
'cytomine_zoom_level' : 1,
'cytomine_dump_type' : 1,
'cytomine_annotation_projects' : [1],  #id of projets from which we dump annotations for learning
'cytomine_predict_terms' : [1], #
'cytomine_excluded_terms' : [2,3], #exclude these term ids
'cytomine_reviewed': True
}


#Parameter values are set through command-line, see test-train.sh
pyxit_parameters = {
'dir_ls' : "/",
#'dir_ts' : "/",
'forest_shared_mem' : False,
#processing
'pyxit_n_jobs' : 10,
#subwindows extraction
'pyxit_n_subwindows' : 100,
'pyxit_min_size' : 0.1,
'pyxit_max_size' : 1.0,
'pyxit_target_width' : 24,  #24x24 
'pyxit_target_height' : 24,
'pyxit_interpolation' : 1,
'pyxit_transpose' : 1, #do we apply rotation/mirroring to subwindows (to enrich training set)
'pyxit_colorspace' : 2, # which colorspace do we use ?
'pyxit_fixed_size' : False, #do we extracted fixed sizes or random sizes (false)
'pyxit_save_to' : '/home/XXX.pkl',
#classifier parameters
'forest_n_estimators' : 10, #number of trees
'forest_max_features' : 28, #number of attributes considered at each node
'forest_min_samples_split' : 1, #nmin
'svm' : 0,
'svm_c': 1.0,
}


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


def main(argv):
    # Define command line options
    p = optparse.OptionParser(description='Pyxit/Cytomine Classification Model Builder',
                              prog='PyXit Classification Model Builder (PYthon piXiT)')

    p.add_option("--cytomine_host", type="string", default = '', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type="string", default = '', dest="cytomine_public_key", help="Cytomine public key")
    p.add_option('--cytomine_private_key',type="string", default = '', dest="cytomine_private_key", help="Cytomine private key")
    p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")	
    p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
    p.add_option('-z', '--cytomine_zoom_level', type='int', dest='cytomine_zoom_level', help="working zoom level")
    p.add_option('--cytomine_dump_type', type='int', dest='cytomine_dump_type', help="annotation type (1=crop, 2=alphamask)")
    p.add_option('--cytomine_annotation_projects', type="string", dest="cytomine_annotation_projects", help="Projects from which annotations are extracted")
    p.add_option('--cytomine_predict_terms', type='string', default='0', dest='cytomine_predict_terms', help="term ids of predicted terms (=positive class in binary mode)")
    p.add_option('--cytomine_excluded_terms', type='string', default='0', dest='cytomine_excluded_terms', help="term ids of excluded terms")
    #p.add_option('--cytomine_reviewed', default=False, action="store_true", dest="cytomine_reviewed", help="Get reviewed annotations only")
    p.add_option('--cytomine_reviewed', type='string', default="False", dest="cytomine_reviewed", help="Get reviewed annotations only")

    p.add_option('--pyxit_target_width', type='int', dest='pyxit_target_width', help="pyxit subwindows width")
    p.add_option('--pyxit_target_height', type='int', dest='pyxit_target_height', help="pyxit subwindows height")
    p.add_option('--pyxit_save_to', type='string', dest='pyxit_save_to', help="pyxit model directory") #future: get it from server db
    p.add_option('--pyxit_colorspace', type='int', dest='pyxit_colorspace', help="pyxit colorspace encoding") #future: get it from server db
    p.add_option('--pyxit_n_jobs', type='int', dest='pyxit_n_jobs', help="pyxit number of jobs for trees") #future: get it from server db
    p.add_option('--pyxit_n_subwindows', default=10, type="int", dest="pyxit_n_subwindows", help="number of subwindows")
    p.add_option('--pyxit_min_size', default=0.5, type="float", dest="pyxit_min_size", help="min size")
    p.add_option('--pyxit_max_size', default=1.0, type="float", dest="pyxit_max_size", help="max size")
    p.add_option('--pyxit_interpolation', default=2, type="int", dest="pyxit_interpolation", help="interpolation method 1,2,3,4")
    #p.add_option('--pyxit_transpose', default=False, action="store_true", dest="pyxit_transpose", help="transpose subwindows")
    p.add_option('--pyxit_transpose', type="string", default="False", dest="pyxit_transpose", help="transpose subwindows")
    #p.add_option('--pyxit_fixed_size', default=False, action="store_true", dest="pyxit_fixed_size", help="extract fixed size subwindows")
    p.add_option('--pyxit_fixed_size', type="string", default="False", dest="pyxit_fixed_size", help="extract fixed size subwindows")
    p.add_option('--forest_n_estimators', default=10, type="int", dest="forest_n_estimators", help="number of base estimators (T)")
    p.add_option('--forest_max_features' , default=1, type="int", dest="forest_max_features", help="max features at test node (k)")
    p.add_option('--forest_min_samples_split', default=1, type="int", dest="forest_min_samples_split", help="minimum node sample size (nmin)")
    p.add_option('--forest_shared_mem', default=False, action="store_true", dest="forest_shared_mem", help="shared mem")
    p.add_option('--svm', default=0, dest="svm", help="final svm classifier: 0=nosvm, 1=libsvm, 2=liblinear, 3=lr-l1, 4=lr-l2", type="int")
    p.add_option('--svm_c', default=1.0, type="float", dest="svm_c", help="svm C")
    #p.add_option('--verbose', action="store_true", default=True, dest="verbose", help="Turn on verbose mode")
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
    parameters['cytomine_dump_type'] = options.cytomine_dump_type

    parameters['cytomine_reviewed'] = str2bool(options.cytomine_reviewed)

    pyxit_parameters['pyxit_target_width'] = options.pyxit_target_width
    pyxit_parameters['pyxit_target_height'] = options.pyxit_target_height
    pyxit_parameters['pyxit_n_subwindows'] = options.pyxit_n_subwindows
    pyxit_parameters['pyxit_min_size'] = options.pyxit_min_size
    pyxit_parameters['pyxit_max_size'] = options.pyxit_max_size
    pyxit_parameters['pyxit_colorspace'] = options.pyxit_colorspace
    pyxit_parameters['pyxit_interpolation'] = options.pyxit_interpolation
    pyxit_parameters['pyxit_transpose'] = str2bool(options.pyxit_transpose)
    pyxit_parameters['pyxit_fixed_size'] = str2bool(options.pyxit_fixed_size)
    pyxit_parameters['forest_n_estimators'] = options.forest_n_estimators
    pyxit_parameters['forest_max_features'] = options.forest_max_features
    pyxit_parameters['forest_min_samples_split'] = options.forest_min_samples_split
    pyxit_parameters['forest_shared_mem'] = options.forest_min_samples_split
    pyxit_parameters['svm'] = options.svm
    pyxit_parameters['svm_c'] = options.svm_c
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
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Run...", progress = 0)



    #Get annotation descriptions (JSON) from project(s)
    annotations = None
    for prj in parameters['cytomine_annotation_projects']:
        if parameters["cytomine_reviewed"]:
            print "Retrieving reviewed annotations..."
            annotations_prj = conn.get_annotations(id_project = prj, reviewed_only = parameters["cytomine_reviewed"])
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
    print "Total annotations projects %s = %d" %(parameters['cytomine_annotation_projects'],len(annotations.data()))
    time.sleep(2)

    
    #Set output dir parameters
    pyxit_parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"], str(parameters['cytomine_annotation_projects']).replace(',','-').replace('[','').replace(']','').replace(' ',''), "zoom_level", str(parameters['cytomine_zoom_level']),"dump_type",str(parameters['cytomine_dump_type']))
    if not os.path.exists(pyxit_parameters['dir_ls']):
        print "Creating annotation directory: %s" %pyxit_parameters['dir_ls']
        os.makedirs(pyxit_parameters['dir_ls'])

        
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Dump annotations...", progress = 50)
    time.sleep(2)
    #Image dump type (for classification use 1)
    if (parameters['cytomine_dump_type']==1):
        annotation_get_func = Annotation.get_annotation_crop_url
    elif (parameters['cytomine_dump_type']==2):
        annotation_get_func = Annotation.get_annotation_alpha_crop_url
    else:
        print "default annotation type crop"
        annotation_get_func = Annotation.get_annotation_crop_url  
    #Dump annotation images locally
    annotations=conn.dump_annotations(annotations = annotations, get_image_url_func = annotation_get_func, dest_path = pyxit_parameters['dir_ls'], desired_zoom = parameters['cytomine_zoom_level'],excluded_terms=parameters['cytomine_excluded_terms'])
    print "END dump annotations"


    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Build model...", progress = 75)
    #Build pyxit classification model (and saves it)
    print "Upload software parameters values to Cytomine-Core..."
    if run_by_user_job==False:
        parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), pyxit_parameters)
        argv = parameters_values_to_argv(pyxit_parameters, parameters_values)
    else:
        argv = []
        for key in pyxit_parameters:
            value = pyxit_parameters[key]
            if type(value) is bool or value == 'True':
                if bool(value):
                    argv.append("--%s" % key)
            elif not value == 'False':
                argv.append("--%s" % key)
                argv.append("%s" % value)


    print "argv :"
    print argv
    print "Run PyXiT..."

    d = os.path.dirname(pyxit_parameters['pyxit_save_to'])
    if not os.path.exists(d):
        os.makedirs(d)

    predict = pyxitstandalone.main(argv)

    print "-------------------------------------------------------"
    print "Pyxit Classification Model saved locally: %s " %pyxit_parameters['pyxit_save_to'] 
    print "-------------------------------------------------------"

    job = conn.update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
