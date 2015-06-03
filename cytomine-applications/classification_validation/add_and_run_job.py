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


import sys
import cytomine
import os
from cytomine.models import *
from cytomine.utils import parameters_values_to_argv

from pyxit import pyxitstandalone
import argparse, optparse
import time


#Examples (default) of parameter values through command-line args
parameters = {
'cytomine_host' : None,
'cytomine_public_key' : None,
'cytomine_private_key' : None,
'cytomine_base_path' : None,
'cytomine_working_path' : '$HOME/tmp/cytomine/annotations/',
'cytomine_id_software' : 816476, #id of the pyxit validation software
'cytomine_id_project' : 716498, #id of the project to which annotation classifications will be uploaded
'cytomine_zoom_level' : 1,
'cytomine_dump_type' : 1, # type of the crop of the annotation (1=full crop, 2=crop with mask)
'cytomine_annotation_projects' : [716498],  #id of projets from which we dump annotations to build the training dataset
'cytomine_excluded_terms' : [676131,676210,676176,], #do not use these cytomine terms 
'cytomine_reviewed': True,
'cytomine_fixed_tile':False,
'cytomine_n_shifts':0,
}


#Examples (default) of parameter values to be set through command-line args
pyxit_parameters = {
'dir_ls' : "/",
'forest_shared_mem' : False,
#processing
'pyxit_n_jobs' : 10,
#subwindows extraction
'pyxit_n_subwindows' : 100,
'pyxit_min_size' : 0.1,
'pyxit_max_size' : 1.0,
'pyxit_target_width' : 16,  #24x24 en zoom 3 sur agar/pgp
'pyxit_target_height' : 16,
'pyxit_interpolation' : 1,
'pyxit_transpose' : 1, #do we apply rotation/mirroring to subwindows (to enrich training set)
'pyxit_colorspace' : 2, # which colorspace do we use ?
'pyxit_fixed_size' : False, #do we extracted fixed sizes or random sizes (false)
'pyxit_save_to' : '/home/maree/tmp/cytomine/models/test.pkl',
#classifier parameters
'forest_n_estimators' : 10, #number of trees
'forest_max_features' : 28, #number of attributes considered at each node
'forest_min_samples_split' : 1, #nmin
'svm' : 0,
'svm_c': 1.0,
#evaluation protocol
'cv_k_folds' : 10,
'cv_shuffle' : False,
'cv_shuffle_test_fraction' : 0.3,
}


def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")


def publish_predict(conn, annotations, y_hat, y_hat_proba):
    i = 0
    for annotation in annotations.data():
        for term in  annotation.term:
            annotation_term = AnnotationTerm()
            annotation_term._callback_identifier = 'algoannotationterm'
            annotation_term.annotation = annotation.id
            annotation_term.expectedTerm = term
            annotation_term.term = int(y_hat[i])
            annotation_term.rate = y_hat_proba[i]
            conn.save(annotation_term)
            i += 1




def main(argv):
    # Define command line options
    p = optparse.OptionParser(description='Pyxit/Cytomine Classification model Builder',
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
    #p.add_option('--cytomine_fixed_tile', default=False,action="store_true", dest="cytomine_fixed_tile", help="Force fixed tile size crop around annotations")
    p.add_option('--cytomine_fixed_tile', type="string", default="False",dest="cytomine_fixed_tile", help="Force fixed tile size crop around annotations")
    p.add_option('--cytomine_n_shifts',type='int', dest='cytomine_n_shifts',help="number of translated (shifted) crops extracted for each annotation")
    p.add_option('--cytomine_annotation_projects', type="string", dest="cytomine_annotation_projects", help="Projects from which annotations are extracted")	
    p.add_option('--cytomine_excluded_terms', type='string', default='5735', dest='cytomine_excluded_terms', help="term ids of excluded terms")
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
    p.add_option('--svm', default=0, dest="svm", help="final svm classifier: 0=nosvm, 1=libsvm, 2=liblinear, 3=lr-l1, 4=lr-l2", type="int")
    p.add_option('--svm_c', default=1.0, type="float", dest="svm_c", help="svm C")
    p.add_option('--cv_k_folds', default=False, type="int", dest="cv_k_folds", help="number of cross validation folds")
    #p.add_option('--cv_shuffle', default=False, action="store_true", dest="cv_shuffle", help="shuffle splits in cross validation")
    p.add_option('--cv_shuffle', type="string", default="False", dest="cv_shuffle", help="shuffle splits in cross validation")
    p.add_option('--cv_shuffle_test_fraction', default=0.3, type="float", dest="cv_shuffle_test_fraction", help="shuffle fraction in cross validation")
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
    parameters['cytomine_excluded_terms'] = map(int,options.cytomine_excluded_terms.split(','))
    parameters['cytomine_zoom_level'] = options.cytomine_zoom_level
    parameters['cytomine_dump_type'] = options.cytomine_dump_type
    parameters['cytomine_fixed_tile'] = str2bool(options.cytomine_fixed_tile)
    parameters['cytomine_n_shifts'] = options.cytomine_n_shifts
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
    pyxit_parameters['svm'] = options.svm
    pyxit_parameters['svm_c'] = options.svm_c
    pyxit_parameters['cv_k_folds'] = options.cv_k_folds
    pyxit_parameters['cv_shuffle'] = str2bool(options.cv_shuffle)
    pyxit_parameters['cv_shuffle_test_fraction'] = options.cv_shuffle_test_fraction
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
    


    #get annotation collection from Cytomine 
    annotations = conn.get_annotations(id_project = parameters['cytomine_id_project'], reviewed_only = parameters['cytomine_reviewed'], id_user=58494)
    #set output dir parameters
    pyxit_parameters['dir_ls'] = os.path.join(parameters["cytomine_working_path"], str(parameters['cytomine_annotation_projects']).replace(',','-').replace('[','').replace(']','').replace(' ',''), "zoom_level", str(parameters['cytomine_zoom_level']),"dump_type",str(parameters['cytomine_dump_type']))
    if not os.path.exists(pyxit_parameters['dir_ls']):
        print "Creating annotation directory: %s" %pyxit_parameters['dir_ls']
        os.makedirs(pyxit_parameters['dir_ls'])
    time.sleep(2)
    #image dump type
    if (parameters['cytomine_dump_type']==1):
        annotation_get_func = Annotation.get_annotation_crop_url
    elif (parameters['cytomine_dump_type']==2):
        annotation_get_func = Annotation.get_annotation_alpha_crop_url
    else:
        print "default annotation type crop"
        annotation_get_func = Annotation.get_annotation_crop_url  
    #dump annotation crops
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Dump annotations...", progress = 50)
    annotations=conn.dump_annotations(annotations = annotations, get_image_url_func = annotation_get_func, dest_path = pyxit_parameters['dir_ls'], desired_zoom = parameters['cytomine_zoom_level'],excluded_terms=parameters['cytomine_excluded_terms'], tile_size = parameters['cytomine_fixed_tile'], translate = parameters['cytomine_n_shifts'])


    #build pyxit model(s) and evaluate them (according to cross-validation parameters)
    print "Create software parameters values..."
    parameters_values = conn.add_job_parameters(user_job.job, conn.get_software(parameters['cytomine_id_software']), pyxit_parameters)
    print "Run PyXiT..."
    argv = parameters_values_to_argv(pyxit_parameters, parameters_values) 
    print argv
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Build models...", progress = 75)
    predict = pyxitstandalone.main(argv)

    
    print "------------------- Publishing results to Cytomine Core... ----------------------"
    print annotations.data()
    job = conn.update_job_status(job, status = job.RUNNING, status_comment = "Publishing results...", progress = 90)


    for annotation in annotations.data():
        #print "ANNOTATION: %s" %annotation
        #print "ANNOTATION TERM: %s" %annotation.term
        for term in annotation.term:
            #annot_descr = conn.get_annotation(annotation.id)
            #if hasattr(annotation, "filename"):
            #    print "filename: %s" %annotation.filename
            #    time.sleep(1)
            if hasattr(annotation, "filename") and (annotation.filename in predict) :
                print "PUBLISH annotation %s prediction" %annotation
                p = predict[annotation.filename]
                annotation_term = AlgoAnnotationTerm()            
                annotation_term.annotation = annotation.id
                annotation_term.expectedTerm = term
                annotation_term.term = p[0]
                if (pyxit_parameters['svm'] == 1): #libsvm does not return proba
                    annotation_term.rate = 1.0
                else:
                    annotation_term.rate = p[1]
                conn.add_annotation_term(annotation.id, p[0], term, annotation_term.rate, annotation_term_model = cytomine.models.AlgoAnnotationTerm)

    job = conn.update_job_status(job, status = job.TERMINATED, status_comment = "Finish", progress = 100)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
