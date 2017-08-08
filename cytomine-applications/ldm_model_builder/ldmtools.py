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


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"

import numpy as np
from scipy import misc
import os


def getcoords(repository, termid):
	if not repository.endswith('/'):
		repository += '/'
	x  = []
	y  = []
	xp = []
	yp = []
	im = []
	for f in os.listdir(repository):
		if f.endswith('.txt'):
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


def getcoordsim(repository, termid, ims):
	if not repository.endswith('/'):
		repository += '/'

	x = []
	y = []
	xp = []
	yp = []
	im = []
	i = 0
	H = {}
	for i in range(len(ims)):
		H[ims[i]]=i

	x = np.zeros(len(ims))
	y = np.zeros(len(ims))
	xp = np.zeros(len(ims))
	yp = np.zeros(len(ims))

	for f in os.listdir(repository):
		if f.endswith('.txt'):
			filename = repository + f
			F = open(filename, 'rb')
			L = F.readlines()
			imageid = int(f.rstrip('.txt'))
			if(imageid in H):
				for j in range(len(L)):
					line = L[j].rstrip()
					v = line.split(' ')
					if (int(v[0]) == termid):
						x[H[imageid]] = int(float(v[1]))
						y[H[imageid]] = int(float(v[2]))
						xp[H[imageid]] = float(v[3])
						yp[H[imageid]] = float(v[4])
			F.close()

	return x, y, xp, yp

def getallcoords(repository):
	if not repository.endswith('/'):
		repository += '/'
	term_to_i = {}
	i_to_term = {}
	nims = len(os.listdir(repository))
	files = os.listdir(repository)
	F = open(repository+files[0])
	lines = F.readlines()
	nldms = len(lines)
	i = 0
	for l in lines:
		v = l.rstrip('\n').split(' ')
		id_term = int(v[0])
		term_to_i[id_term] = i
		i_to_term[i] = id_term
		i += 1

	F.close()

	X = np.zeros((nims,nldms))
	Y = np.zeros((nims,nldms))
	Xr = np.zeros((nims,nldms))
	Yr = np.zeros((nims,nldms))

	ims = []
	im = 0
	for f in os.listdir(repository):
		filename = repository+f
		F = open(filename,'rb')
		L = F.readlines()
		for l in L:
			v = l.rstrip().split(' ')
			id_term = int(v[0])

			X[im,term_to_i[id_term]] = float(v[1])
			Y[im,term_to_i[id_term]] = float(v[2])
			Xr[im,term_to_i[id_term]] = float(v[3])
			Yr[im,term_to_i[id_term]] = float(v[4])
		F.close()
		ims.append(int(f.rstrip('.txt')))
		im+=1

	return X,Y,Xr,Yr,ims,term_to_i,i_to_term


def readimage(repository,idimage,image_type='jpg'):
	if not repository.endswith('/'):
		repository += '/'

	if image_type=='png':
		IM = misc.imread('%s%d.png'%(repository,idimage),flatten=True)
	elif image_type=='bmp':
		IM = misc.imread('%s%d.bmp'%(repository,idimage),flatten=True)
	elif image_type=='jpg':
		IM = misc.imread('%s%d.jpg'%(repository,idimage),flatten=True)

	IM = np.double(IM)
	IM -= np.mean(IM)
	IM  /= np.std(IM)

	return IM


def makesize(IM, wp):
	(h, w) = IM.shape
	IM2 = np.zeros((h + 2 * wp, w + 2 * wp))
	IM2[wp:wp + h, wp:wp + w] = IM
	return IM2

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")
