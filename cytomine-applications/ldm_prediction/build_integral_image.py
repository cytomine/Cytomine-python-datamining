# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2017. Authors: see NOTICE file.
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
__copyright__       = "Copyright 2010-2017 University of Liège, Belgium, http://www.cytomine.be/"

import numpy as np
import scipy.misc as misc
from multiprocessing import Pool


def build_integral_image_from_path(path_to_image):
	img = misc.imread(path_to_image,flatten=True)
	img = img.astype(float)/255.
	i_img = build_integral_image(img)
	
	for ext in ['bmp','png','jpg']:
		path_to_image = path_to_image.rstrip('.%s'%ext)
	path_to_image += '_integral'
	np.save(path_to_image,i_img)


def build_integral_image(img):
	(h,w) = img.shape
	i_img = np.zeros((h,w))
	
	i_img[0,0] = img[0,0]
	for i in range(1,h):
		i_img[i,0] = i_img[i-1,0]+img[i,0]
	for i in range(1,w):
		i_img[0,i] = i_img[0,i-1]+img[0,i]
	for i in range(1,h):
		for j in range(1,w):
			i_img[i,j] = img[i,j]+i_img[i-1,j]+i_img[i,j-1]-i_img[i-1,j-1]
	return i_img


def build_integral_images_mp(tab_path,n_jobs):
	p = Pool(n_jobs)
	p.map(build_integral_image_from_path,tab_path)
	p.close()
	p.join()

