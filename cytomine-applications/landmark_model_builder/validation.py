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


__author__          = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import sys,optparse
import numpy as np
from scipy import misc
import scipy.ndimage as snd
from sklearn.ensemble import ExtraTreesClassifier
import pickle
import os
from multiprocessing import Pool
import scipy
from array import *
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import cytomine
from cytomine import cytomine, models
from cytomine.models import *
from download import *

"""
This function extracts the coordinates of a given term from an offline
cytomine images/coordinates repository.
"""
def getcoords(repository,termid):
	if(not repository.endswith('/')):
		repository = repository+'/'
	liste = os.listdir(repository)
	
	x  = []
	y  = []
	xp = []
	yp = []
	im = []
	
	for f in os.listdir(repository):
		if(f.endswith('.txt')):
			filename = repository+f
			F = open(filename,'rb')
			L = F.readlines()
			imageid = int(f.rstrip('.txt'))
			for j in range(len(L)):
				line = L[j].rstrip()
				v = line.split(' ')
				if(int(v[0])==termid):
					x.append(int(float(v[1])))
					y.append(int(float(v[2])))
					xp.append(float(v[3]))
					yp.append(float(v[4]))
					im.append(imageid)
			F.close()
	return np.array(x),np.array(y),np.array(xp),np.array(yp),np.array(im)

"""
This function returns a 8 bit grey-value image given its identifier in the 
offline cytomine repository.
"""	
def readimage(repository,idimage,image_type='jpg'):
	if(not repository.endswith('/')):
		repository = repository+'/'

	if(image_type=='png'):
		IM = misc.imread('%s%d.png'%(repository,idimage),flatten=True)
	elif(image_type=='bmp'):
		IM = misc.imread('%s%d.bmp'%(repository,idimage),flatten=True)
	elif(image_type=='jpg'):
		IM = misc.imread('%s%d.jpg'%(repository,idimage),flatten=True)
	IM = np.double(IM)
	IM = IM-np.mean(IM)
	IM = IM/np.std(IM)
	return IM

"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""
def searchpoint(repository,current,clf,mx,my,cm,depths,window_size,image_type,
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
				
	return np.median(maxx),np.median(maxy),height-np.median(maxy)

"""
0-padding of an image IM of wp pixels on all sides
"""
def makesize(IM,wp):
	(h,w) = IM.shape
	IM2 = np.zeros((h+2*wp,w+2*wp))
	IM2[wp:wp+h,wp:wp+w] = IM
	return IM2

"""
Build the dataset on a single image
"""
def build_dataset_image(IM,wp,x_v,y_v,depths):
	
	swp = (2*wp)**2
	wp1 = wp+1
	ndata=len(x_v)	
	dwp=2*wp
	ndepths = len(depths)
	
	DATASET = np.zeros((ndata,swp*ndepths),'uint8')
	REP = np.zeros(ndata)
	
	images = {}
	for z in xrange(ndepths):
		images[z] = makesize(snd.zoom(IM,depths[z]),wp)
		
	X = [[int(x*depths[z]) for x in x_v] for z in xrange(ndepths)]
	Y = [[int(y*depths[z]) for y in y_v] for z in xrange(ndepths)]
	cub = np.zeros((ndepths,dwp,dwp))
	
	for j in xrange(ndata):
		x = x_v[j]		
		y = y_v[j]
		for z in xrange(ndepths):
			im = images[z]
			cub[z,:,:] = im[Y[z][j]:Y[z][j]+dwp,X[z][j]:X[z][j]+dwp]
		DATASET[j,:] = cub.flatten()
	
	return DATASET

def rotate_coordinates(repository,num,x,y,angle,image_type):
	image = readimage(repository,num,image_type)
	image_rotee = snd.rotate(image,-angle)
	(h,w) = image.shape
	(hr,wr) = image_rotee.shape
	angle_rad = angle*(np.pi/180.)
	c = np.cos(angle_rad)
	s = np.sin(angle_rad)
	x = x-(w/2.)
	y = y-(h/2.)
	xr = ((x*c)-(y*s))+(wr/2.)
	yr = ((x*s)+(y*c))+(hr/2.)

	return xr.tolist(),yr.tolist(),image_rotee
	
def dataset_image_rot(repository,Xc,Yc,R,RMAX,proportion,step,i,ang,depths,window_size,image_type):
		#print i
		x_v = []
		y_v = []
		REP = []
		IMGS = []
		deuxpi = 2.*np.pi
		for x in np.arange(np.int(Xc)-R,np.int(Xc)+R+1,step):
			for y in np.arange(np.int(Yc)-R,np.int(Yc)+R+1,step):
				if(np.linalg.norm([Xc-x,Yc-y])<=R):
					x_v.append(x)
					y_v.append(y)
					REP.append(1)
					IMGS.append(i)
		
		n = len(x_v)
		image = readimage(repository,i,image_type)
		(height,width)=image.shape
		height=height-1
		width=width-1
		for t in range(int(round(proportion*n))):
			angle = np.random.ranf()*deuxpi
			r = R + (np.random.ranf()*(RMAX-R))
			tx = int(r*np.cos(angle))
			ty = int(r*np.sin(angle))
			x_v.append(min(width,Xc+tx))
			y_v.append(min(height,Yc+ty))
			REP.append(0)
			IMGS.append(i)
			
		(x_r,y_r,im) = rotate_coordinates(repository,i,np.round(np.array(x_v)),np.round(np.array(y_v)),angle,image_type)
		(hr,wr) = im.shape
		hr = hr-1
		wr = wr-1
		
		x_r = np.round(x_r)
		y_r = np.round(y_r)
		
		for index in range(len(x_r)):
			xr = x_r[index]
			yr = y_r[index]
			if(xr<0):
				x_r[index] = 0
			if(yr<0):
				y_r[index] = 0
			if(xr>wr):
				x_r[index] = wr
			if(yr>hr):
				y_r[index] = hr			
		return build_dataset_image(im,window_size,x_r,y_r,depths),REP,IMGS

def mp_helper_rot(job_args):
	return dataset_image_rot(*job_args)
	
def build_datasets_rot_mp(repository,imc,Xc,Yc,R,RMAX,proportion,step,ang,window_size,depths,nimages,image_type, njobs):
	TOTDATA = None
	TOTREP = None
	IMGS = []
	X = None
	Y = None
	deuxpi = 2.*np.pi

	p = Pool(njobs) 
	job_args = [(repository, Xc[i], Yc[i], R, RMAX, proportion, step, imc[i], (np.random.ranf()*2*ang)-ang, depths, window_size, image_type) for i in range(nimages)] 
	T = p.map(mp_helper_rot,job_args)
	p.close()
	p.join()
	return T

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
	
if __name__ == "__main__":
	
	parameters = {
		'cytomine_host' : '',
		'cytomine_public_key' : '',
		'cytomine_private_key' : '',
		'cytomine_id_software': 0,
		'cytomine_base_path' : '',
		'cytomine_working_path' : None,
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
	p.add_option('--validation_K', type='int', default=10, dest='validation_K', help="K for K-fold cross validation")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")
	
	options, arguments = p.parse_args( args = sys.argv)

	parameters['cytomine_host'] = options.cytomine_host	
	parameters['cytomine_public_key'] = options.cytomine_public_key
	parameters['cytomine_private_key'] = options.cytomine_private_key
	parameters['cytomine_id_software'] = options.cytomine_id_software
	parameters['cytomine_base_path'] = options.cytomine_base_path
	parameters['cytomine_working_path'] = options.cytomine_working_path
	parameters['cytomine_id_term'] = options.cytomine_id_term
	parameters['cytomine_id_project'] = options.cytomine_id_project
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
	parameters['validation_K'] = options.validation_K
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
	job = cytomine_connection.update_job_status(job, status = job.RUNNING, progress = 0, status_comment = "Beginning validation...")
	
	download_images(cytomine_connection,parameters['cytomine_id_project'])
	download_annotations(cytomine_connection,parameters['cytomine_id_project'],parameters['cytomine_working_path'])
		
	repository = parameters['cytomine_working_path']+str(parameters['cytomine_id_project'])+'/'
	txt_repository = parameters['cytomine_working_path']+'%d/txt/'%parameters['cytomine_id_project']
	
	depths = 1./(2.**np.arange(parameters['model_depth']))
	image_type = parameters['image_type']

	
	(xc,yc,xr,yr,imc) = getcoords(txt_repository,parameters['cytomine_id_term'])
	nimages = np.max(xc.shape)
	mx = np.mean(xr)
	my = np.mean(yr)
	P = np.zeros((2,nimages))
	P[0,:] = xr
	P[1,:] = yr
	cm = np.cov(P)
	
	passe = False
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
				DATA = np.zeros((height*(len(T)+100)*parameters['model_ntimes'],width),'uint8')
				REP = np.zeros(height*(len(T)+100)*parameters['model_ntimes'],'uint8')
				IMG = np.zeros(height*(len(T)+100)*parameters['model_ntimes'])
				b=0
				be=height
			DATA[b:be,:]=data
			REP[b:be]=rep
			IMG[b:be]=img
			b=be
			be=be+height
	REP = REP[0:b]
	DATA = DATA[0:b,:]
	IMG = IMG[0:b]
	erreur = []
	g = np.random.randint(0,parameters['validation_K'],nimages)
	groupes = np.zeros(IMG.shape)
	G = {}	
	for i in range(parameters['validation_K']):
		G[i]=[]
	for i in range(nimages):
		t = np.where(IMG==imc[i])
		t = t[0]
		groupes[t] = g[i]
		G[g[i]].append(imc[i])
	
	Xh = {}
	Yh = {}
	for i in range(nimages):
		Xh[imc[i]]=xc[i]
		Yh[imc[i]]=yc[i]
		
	for k in range(parameters['validation_K']):
		t = np.where(groupes!=k)
		t = t[0]
		TRDATA = DATA[t,:]
		TRREP = REP[t]
		clf = ExtraTreesClassifier(n_jobs=parameters['model_njobs'],n_estimators=parameters['model_ntrees'])
		clf = clf.fit(TRDATA,TRREP)
		for j in G[k]:
			(x,y,yp) = searchpoint(repository,j,clf,mx,my,cm,depths,parameters['model_wsize'],parameters['image_type'],parameters['model_npred'])
			circle = Point(x,yp)
			location = circle.wkt
			new_annotation = cytomine_connection.add_annotation(location, j)
			cytomine_connection.add_annotation_term(new_annotation.id, term=parameters['cytomine_id_term'],expected_term=parameters['cytomine_id_term'],rate=1.0,annotation_term_model = models.AlgoAnnotationTerm)
			er = np.linalg.norm([x-Xh[j],y-Yh[j]])
			print j,x,y,Xh[j],Yh[j],er
			erreur.append(er)

	moy = np.mean(erreur)
	ste = np.std(erreur)
	
	ec95 = 1.96*(np.std(erreur)/np.sqrt(nimages))
	print parameters['cytomine_id_term'],moy-ec95,moy,moy+ec95
	
	job_parameters= {}
	job_parameters['landmark_term'] = parameters['cytomine_id_term']
	job_parameters['landmark_r'] = parameters['model_R']
	job_parameters['landmark_rmax'] = parameters['model_RMAX']
	job_parameters['landmark_p'] = parameters['model_P']
	job_parameters['landmark_npred'] = parameters['model_npred']
	job_parameters['landmark_ntimes'] = parameters['model_ntimes']
	job_parameters['landmark_alpha'] = parameters['model_angle']
	job_parameters['landmark_depth'] = parameters['model_depth']
	job_parameters['landmark_window_size'] = parameters['model_wsize']
	job_parameters['forest_n_estimators'] = parameters['model_ntrees']
	job_parameters['forest_max_features'] = 10
	job_parameters['forest_min_samples_split'] = 2
	job_parameters['validation_K'] = parameters['validation_K']
	job_parameters['validation_result_mean'] = moy
	
	if run_by_user_job==False:
	    job_parameters_values = cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(parameters['cytomine_id_software']),job_parameters)
	job = cytomine_connection.update_job_status(job, status = job.TERMINATED, progress = 100, status_comment = "Validation done.")
