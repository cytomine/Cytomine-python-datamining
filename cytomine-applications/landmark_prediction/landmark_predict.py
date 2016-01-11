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

from sklearn.externals import joblib
from array import *
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import sys,optparse
sys.path.append('/software_router/algo/landmark_model_builder/')
from validation import *
from download import *
import cytomine
from cytomine import cytomine, models
from cytomine.models import *
"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""
def searchpoint_cytomine(repository,current,clf,mx,my,cm,depths,window_size,image_type,
	npred):
	simage = readimage(repository,current,image_type)
	(height,width) = simage.shape
	
	P = np.random.multivariate_normal([mx,my],cm,npred)	
	x_v = np.round(P[:,0]*width)
	y_v = np.round(P[:,1]*height)
	
	height=height-1
	width=width-1

	n = len(x_v)
	pos = 0
	
	maxprob = -1
	maxx = []
	maxy = []
	
	#maximum number of points considered at once in order to not overload the
	#memory.
	step = 100000

	for index in range(len(x_v)):
		xv = x_v[index]
		yv = y_v[index]
		if(xv<0):
			x_v[index] = 0
		if(yv<0):
			y_v[index] = 0
		if(xv>width):
			x_v[index] = width
		if(yv>height):
			y_v[index] = height
	
	while(pos<n):
		xp = np.array(x_v[pos:min(n,pos+step)])
		yp = np.array(y_v[pos:min(n,pos+step)])
		
		DATASET = build_dataset_image(simage,window_size,xp,yp,depths)
		pred = clf.predict_proba(DATASET)
		pred = pred[:,1]
		maxpred = np.max(pred)
		if(maxpred>=maxprob):
			positions = np.where(pred==maxpred)
			positions = positions[0]
			xsup = xp[positions]
			ysup = yp[positions]
			if(maxpred>maxprob):
				maxprob = maxpred
				maxx = xsup
				maxy = ysup
			else:
				maxx = np.concatenate((maxx,xsup))
				maxy = np.concatenate((maxy,ysup))
		pos = pos+step
				
	return np.median(maxx),(height+1)-np.median(maxy)

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
		'cytomine_id_project' : 0,
		'model_load_from':'',
		'cytomine_model_names' : '',
		'image_type': '',
		'verbose': False
	}
	
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Landmark Detection',prog='Cytomine Landmark Detection : Landmark Dectector',version='0.1')
	p.add_option('--cytomine_host', type="string", default = 'beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default = 'XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key',type="string", default = 'YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default = '/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")    
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--model_load_from', default='/tmp/', type="string", dest="model_load_from", help="The repository where the models are stored")
	p.add_option('--model_names', type="string", dest="model_names", help="The names of the models to use for detection (separated by commas, no spaces)")
	p.add_option('--image_type', type='string', default= 'jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")
	
	options, arguments = p.parse_args( args = sys.argv)

	parameters['cytomine_host'] = options.cytomine_host	
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path	
	parameters['cytomine_id_project'] = options.cytomine_id_project	
	parameters['model_load_from'] = options.model_load_from
	parameters['model_names'] = options.model_names		
	parameters['image_type'] = options.image_type
	parameters['verbose'] = str2bool(options.verbose)
	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'],parameters['cytomine_public_key'],parameters['cytomine_private_key'],base_path=parameters['cytomine_base_path'],working_path=parameters['cytomine_working_path'],verbose=parameters['verbose'])
	
	if(not parameters['cytomine_working_path'].endswith('/')):
		parameters['cytomine_working_path'] = parameters['cytomine_working_path']+'/'
	if(not parameters['model_load_from'].endswith('/')):
		parameters['model_load_from'] = parameters['model_load_from']+'/'
		
	download_images(cytomine_connection,int(parameters['cytomine_id_project']))
	
	repository = parameters['cytomine_working_path']+str(parameters['cytomine_id_project'])+'/'
	
	model_repo = parameters['model_load_from']
	model_names = parameters['model_names'].split(',')
	nmodels = len(model_names)
	image_type = parameters['image_type']	
	id_software = parameters['cytomine_id_software']
	coords = {}
	ips = []
	Rs = []
	RMAXs = []
	proportions = []
	npreds = []
	ntreess = []
	ntimess = []
	angranges = []
	depthss = []
	steps = []
	window_sizes = []
	max_features = []
	min_samples = []
	n = 0

	current_user = cytomine_connection.get_current_user()
	run_by_user_job = False
	if current_user.algo==False:
	    user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
	    cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
	    user_job = current_user
	    run_by_user_job = True

	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Uploading annotations...")
	job_parameters= {}
	job_parameters['landmark_terms'] = ips
	job_parameters['model_id_job'] = 0
	job_parameters['landmark_r'] = Rs
	job_parameters['landmark_rmax'] = RMAXs
	job_parameters['landmark_p'] = proportions
	job_parameters['landmark_npred'] = npreds
	job_parameters['landmark_ntimes'] = ntimess
	job_parameters['landmark_alpha'] = angranges
	job_parameters['landmark_depth'] = depthss
	job_parameters['landmark_window_size'] = window_sizes
	job_parameters['forest_n_estimators'] = ntreess
	job_parameters['forest_max_features'] = max_features
	job_parameters['forest_min_samples_split'] = min_samples

	if run_by_user_job==False:
	    job_parameters_values = cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(id_software),job_parameters)

	progress = 0
	delta = 90/len(model_names)

	for model in model_names:
		F = open('%s%s.conf'%(model_repo,model))
		par = {}
		for l in F.readlines():
			line = l.rstrip('\n')
			tab = line.split(' ')
			par[tab[0]] = float(tab[1])
		
		ips.append(int(par['cytomine_id_term']))
		Rs.append(par['model_R'])
		RMAXs.append(par['model_RMAX'])
		proportions.append(par['model_P'])
		npreds.append(int(par['model_npred']))
		ntreess.append(int(par['model_ntrees']))
		ntimess.append(int(par['model_ntimes']))
		angranges.append(par['model_angle'])
		depthss.append(int(par['model_depth']))
		steps.append(par['model_step'])
		window_sizes.append(int(par['window_size']))
		max_features.append(int(np.sqrt(((int(par['window_size'])*2)**2)*int(par['model_depth']))))
		min_samples.append(2)
		mx,my,cm = joblib.load('%s%s_cov.pkl'%(model_repo,model))
		clf = joblib.load('%s%s.pkl'%(model_repo,model))

		progress += delta
		job = cytomine_connection.update_job_status(job, status = job.RUNNING, status_comment = "Analyzing models", progress = progress)

		for f in os.listdir(repository):
			if(f.endswith('.%s'%image_type)):
				tab = f.split('.%s'%image_type)
				j = int(tab[0])
				(x,y) = searchpoint_cytomine(repository,j,clf,mx,my,cm,1./(2.**np.arange(depthss[n])),window_sizes[n],image_type,npreds[n])
				circle = Point(x,y)
				location = circle.wkt
				new_annotation = cytomine_connection.add_annotation(location, j)
				cytomine_connection.add_annotation_term(new_annotation.id, term=ips[n],expected_term=ips[n],rate=1.0,annotation_term_model = models.AlgoAnnotationTerm)
		n = n+1
	
	job = cytomine_connection.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment =  "Annotations uploaded!")
	print "Annotations uploaded!"
