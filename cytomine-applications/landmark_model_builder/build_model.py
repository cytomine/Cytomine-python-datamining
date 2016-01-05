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


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from validation import *
from sklearn.externals import joblib
import optparse,sys



def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
		
if __name__ == "__main__":

	parameters = {
		'cytomine_host' : '',
		'cytomine_public_key' : '',
		'cytomine_private_key' : '',
		'cytomine_id_software': 0,
		'cytomine_base_path' : '',
		'cytomine_working_path' : '',
		'cytomine_id_term': None,
		'cytomine_id_project': None,
		'image_type': '',
		'model_njobs': None,
		'model_R': None,
		'model_RMAX': None,
		'model_P': None,
		'model_npred': None,
		'model_ntrees': None,
		'model_ntimes': None,
		'model_angle': None,
		'model_depth': None,
		'model_step': None,
		'model_wsize': None,
		'model_name': '',
		'model_save_to':'',
		'verbose': False
	}
	
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Model building',prog='Cytomine Landmark Detector : Model builder',version='0.1')
	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = 'XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = 'YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")	
	p.add_option('--cytomine_id_term', type='int', dest='cytomine_id_term', help="The identifier of the term to create a detection model for")
	p.add_option('--image_type', type='string', default= 'jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--model_njobs', type='int', default=1, dest='model_njobs', help="The number of processors used for model building")
	p.add_option('--model_R', type='int', default=6, dest='model_R', help="Max distance for extracting landmarks")
	p.add_option('--model_RMAX', type='int', default=200, dest='model_RMAX', help="Max distance for extracting non-landmarks")
	p.add_option('--model_P', type='float', default=3, dest='model_P', help="Proportion of non-landmarks")
	p.add_option('--model_npred', type='int', default=50000, dest='model_npred', help="Number of pixels extracted for prediction")
	p.add_option('--model_ntrees', type='int', default=50, dest='model_ntrees', help="Number of trees")
	p.add_option('--model_ntimes', type='int', default=3, dest='model_ntimes', help="Number of rotations to apply to the image")
	p.add_option('--model_angle', type='float', default=30, dest='model_angle', help="Max angle for rotation")
	p.add_option('--model_depth', type='int', default=5, dest='model_depth', help="Number of resolutions to use")
	p.add_option('--model_step', type='int', default=1, dest='model_step', help="Landmark pixels will be extracted in a grid (x-R:step:x+r,y-R:step:y+R) around the landmark")
	p.add_option('--model_wsize', type='int',default=8, dest='model_wsize', help="Window size")
	p.add_option('--model_save_to', type='string', default='/tmp/', dest='model_save_to', help="Destination for model storage")
	p.add_option('--model_name', type='string', dest='model_name', help="Name of the model (used for saving)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")
	
	options, arguments = p.parse_args( args = sys.argv)
	
	
	parameters['cytomine_host'] = options.cytomine_host	
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_id_project'] = options.cytomine_id_project
	parameters['cytomine_id_term'] = options.cytomine_id_term
	parameters['image_type'] = options.image_type
	parameters['model_njobs'] = options.model_njobs
	parameters['model_R'] = options.model_R
	parameters['model_RMAX'] = options.model_RMAX
	parameters['model_P'] = options.model_P
	parameters['model_npred'] = options.model_npred
	parameters['model_ntrees'] = options.model_ntrees
	parameters['model_ntimes'] = options.model_ntimes
	parameters['model_angle'] = options.model_angle	
	parameters['model_depth'] = options.model_depth
	parameters['model_step'] = options.model_step
	parameters['model_wsize'] = options.model_wsize
	parameters['model_save_to'] = options.model_save_to
	parameters['model_name'] = options.model_name
	parameters['verbose'] = str2bool(options.verbose)
	
	if(not parameters['cytomine_working_path'].endswith('/')):
		parameters['cytomine_working_path'] = parameters['cytomine_working_path']+'/'
	
	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'],parameters['cytomine_public_key'],parameters['cytomine_private_key'],base_path=parameters['cytomine_base_path'],working_path=parameters['cytomine_working_path'],verbose=parameters['verbose'])

	current_user = cytomine_connection.get_current_user()
	run_by_user_job = False
	if current_user.algo==False:
	    user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
	    cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
	    user_job = current_user
	    run_by_user_job = True

	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Bulding model...")
	job_parameters= {}
	job_parameters['cytomine_id_term'] = parameters['cytomine_id_term']
	job_parameters['model_R'] = parameters['model_R']
	job_parameters['model_njobs'] = parameters['model_njobs']
	job_parameters['model_RMAX'] = parameters['model_RMAX']
	job_parameters['model_P'] = parameters['model_P']
	job_parameters['model_npred'] = parameters['model_npred']
	job_parameters['model_ntimes'] = parameters['model_ntimes']
	job_parameters['model_angle'] = parameters['model_angle']
	job_parameters['model_depth'] = parameters['model_depth']
	job_parameters['model_wsize'] = parameters['model_wsize']
	job_parameters['model_ntrees'] = parameters['model_ntrees']
	job_parameters['model_step'] = parameters['model_step']
	job_parameters['forest_max_features'] = ((2*parameters['model_wsize'])**2)*parameters['model_depth']
	job_parameters['forest_min_samples_split'] = 2

	if run_by_user_job==False:
	    job_parameters_values = cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(parameters['cytomine_id_software']),job_parameters)

	download_images(cytomine_connection,parameters['cytomine_id_project'])
	download_annotations(cytomine_connection,parameters['cytomine_id_project'],parameters['cytomine_working_path'])
			
	repository = parameters['cytomine_working_path']+str(parameters['cytomine_id_project'])+'/'
	txt_repository = parameters['cytomine_working_path']+'%d/txt/'%parameters['cytomine_id_project']

	depths = 1./(2.**np.arange(parameters['model_depth']))

	(xc,yc,xr,yr,imc) = getcoords(txt_repository,parameters['cytomine_id_term'])
	nimages = np.max(xc.shape)
	mx = np.mean(xr)
	my = np.mean(yr)
	P = np.zeros((2,nimages))
	P[0,:] = xr
	P[1,:] = yr
	cm = np.cov(P)
	
	passe = False

	progress = 0
	delta = 80/len(images)

	for times in range(parameters['model_ntimes']):
		if(times==0):
			rangrange=0
		else:
			rangrange = parameters['model_angle']
		T = build_datasets_rot_mp(repository,imc,xc,yc,parameters['model_R'],parameters['model_RMAX'],parameters['model_P'],parameters['model_step'],rangrange,parameters['model_wsize'],depths,nimages,parameters['image_type'],parameters['model_njobs'])
		for i in range(len(T)):
			(data,rep,img) = T[i]
			(height,width) = data.shape
			if(not passe):
				passe = True
				DATA = np.zeros((height*(len(T)+100)*parameters['model_ntimes'],width))
				REP = np.zeros(height*(len(T)+100)*parameters['model_ntimes'])
				b=0
				be=height
			DATA[b:be,:]=data
			REP[b:be]=rep
			b=be
			be=be+height

		progress += delta
		job = cytomine_connection.update_job_status(job, status = job.RUNNING, status_comment = "Bulding model...", progress = progress)

	REP = REP[0:b]
	DATA = DATA[0:b,:]
	#IMG = IMG[0:b]
	clf = ExtraTreesClassifier(n_jobs=parameters['model_njobs'],n_estimators=parameters['model_ntrees'])
	clf = clf.fit(DATA,REP)

	job = cytomine_connection.update_job_status(job, status = job.RUNNING, progress = 90, status_comment = "Writing model...")

	model_repo = parameters['model_save_to']
	if(not os.path.isdir(model_repo)):
		os.mkdir(model_repo)

	joblib.dump(clf,'%s%s.pkl'%(model_repo,parameters['model_name']))
	joblib.dump([mx,my,cm],'%s%s_cov.pkl'%(model_repo,parameters['model_name']))
	F = open('%s%s.conf'%(model_repo,parameters['model_name']),'w+')
	F.write('cytomine_id_term %d\n'%parameters['cytomine_id_term'])
	F.write('model_R %d\n'%parameters['model_R'])
	F.write('model_RMAX %d\n'%parameters['model_RMAX'])
	F.write('model_P %f\n'%parameters['model_P'])
	F.write('model_npred %d\n'%parameters['model_npred'])
	F.write('model_ntrees %d\n'%parameters['model_ntrees'])
	F.write('model_ntimes %d\n'%parameters['model_ntimes'])
	F.write('model_angle %f\n'%parameters['model_angle'])
	F.write('model_depth %d\n'%parameters['model_depth'])
	F.write('model_step %d\n'%parameters['model_step'])
	F.write('window_size %d'%parameters['model_wsize'])
	F.close()
	job = cytomine_connection.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Model built!")
	print "Model built!"
