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


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from ldmtools import *
from sklearn.externals import joblib
import optparse, sys
from sklearn.ensemble import ExtraTreesClassifier
import scipy.ndimage as snd
from multiprocessing import Pool
import cytomine
from download import *

def build_integral_slice(img):
	img = np.double(img)
	img -= np.mean(img)
	img /= np.std(img)
	intg = np.zeros(img.shape)
	(h, w) = img.shape
	intg[0, 0] = img[0, 0]
	for i in range(1, w):
		intg[0, i] = intg[0, i - 1] + img[0, i]
	for i in range(1, h):
		intg[i, 0] = intg[i - 1, 0] + img[i, 0]
	for i in range(1, h):
		i1 = i - 1
		for j in range(1, w):
			j1 = j - 1
			intg[i, j] = img[i, j] + intg[i1, j] + intg[i, j1] - intg[i1, j1]
	return intg


def generate_2_horizontal(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w, n)
	coords[:, 1] = np.random.randint(-w, w + 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(1, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(0, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_2_vertical(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w + 1, n)
	coords[:, 1] = np.random.randint(-w, w, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(0, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(1, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_3_horizontal(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w - 1, n)
	coords[:, 1] = np.random.randint(-w, w + 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(2, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(0, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_3_vertical(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w + 1, n)
	coords[:, 1] = np.random.randint(-w, w - 1, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(0, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(2, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_square(w, n):
	coords = np.zeros((n, 4))
	coords[:, 0] = np.random.randint(-w, w, n)
	coords[:, 1] = np.random.randint(-w, w, n)
	coords[:, 2] = [coords[i, 0] + np.random.randint(1, w + 1 - coords[i, 0]) for i in range(n)]
	coords[:, 3] = [coords[i, 1] + np.random.randint(1, w + 1 - coords[i, 1]) for i in range(n)]
	return coords


def generate_2d_coordinates_horizontal(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 6))
	y = np.zeros((n, 6))

	w = np.floor(0.5 * ((coords[:, 2] - coords[:, 0]) + 1.)).astype('int')
	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = x[:, 0] + w
	y[:, 1] = y[:, 0]
	x[:, 2] = coords[:, 2]
	y[:, 2] = y[:, 1]
	x[:, 3] = x[:, 0]
	y[:, 3] = coords[:, 3]
	x[:, 4] = x[:, 1]
	y[:, 4] = y[:, 3]
	x[:, 5] = x[:, 2]
	y[:, 5] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_2d_coordinates_vertical(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 6))
	y = np.zeros((n, 6))

	w = np.floor(0.5 * ((coords[:, 3] - coords[:, 1]) + 1)).astype('int')
	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = coords[:, 2]
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 0]
	y[:, 2] = y[:, 0] + w
	x[:, 3] = x[:, 1]
	y[:, 3] = y[:, 2]
	x[:, 4] = x[:, 2]
	y[:, 4] = coords[:, 3]
	x[:, 5] = x[:, 3]
	y[:, 5] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_3d_coordinates_horizontal(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 8))
	y = np.zeros((n, 8))
	w = np.floor(((coords[:, 2] - coords[:, 0]) + 1.) / 3.).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = x[:, 0] + w
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 1] + w
	y[:, 2] = y[:, 0]
	x[:, 3] = coords[:, 2]
	y[:, 3] = y[:, 0]
	x[:, 4] = x[:, 0]
	y[:, 4] = coords[:, 3]
	x[:, 5] = x[:, 1]
	y[:, 5] = y[:, 4]
	x[:, 6] = x[:, 2]
	y[:, 6] = y[:, 4]
	x[:, 7] = x[:, 3]
	y[:, 7] = y[:, 4]

	return x.astype('int'), y.astype('int')


def generate_3d_coordinates_vertical(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 8))
	y = np.zeros((n, 8))
	w = np.floor(((coords[:, 3] - coords[:, 1]) + 1.) / 3.).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1
	x[:, 1] = coords[:, 2]
	y[:, 1] = y[:, 0]
	x[:, 2] = x[:, 0]
	y[:, 2] = y[:, 0] + w
	x[:, 3] = x[:, 1]
	y[:, 3] = y[:, 2]
	x[:, 4] = x[:, 2]
	y[:, 4] = y[:, 2] + w
	x[:, 5] = x[:, 3]
	y[:, 5] = y[:, 4]
	x[:, 6] = x[:, 4]
	y[:, 6] = coords[:, 3]
	x[:, 7] = x[:, 5]
	y[:, 7] = y[:, 6]

	return x.astype('int'), y.astype('int')


def generate_square_coordinates(coords):
	(n, quatre) = coords.shape
	x = np.zeros((n, 9))
	y = np.zeros((n, 9))

	wx = np.floor(0.5 * ((coords[:, 2] - coords[:, 0]) + 1.)).astype('int')
	wy = np.floor(0.5 * ((coords[:, 3] - coords[:, 1]) + 1.)).astype('int')

	x[:, 0] = coords[:, 0] - 1
	y[:, 0] = coords[:, 1] - 1

	x[:, 1] = x[:, 0] + wx
	y[:, 1] = y[:, 0]

	x[:, 2] = coords[:, 2]
	y[:, 2] = y[:, 0]

	x[:, 3] = x[:, 0]
	y[:, 3] = y[:, 0] + wy

	x[:, 4] = x[:, 1]
	y[:, 4] = y[:, 3]

	x[:, 5] = x[:, 2]
	y[:, 5] = y[:, 4]

	x[:, 6] = x[:, 3]
	y[:, 6] = coords[:, 3]

	x[:, 7] = x[:, 4]
	y[:, 7] = y[:, 6]

	x[:, 8] = x[:, 5]
	y[:, 8] = y[:, 6]

	return x.astype('int'), y.astype('int')


def build_dataset_image(IM, wp, x_v, y_v, feature_type, feature_parameters, depths):
	feature_type = feature_type.lower()
	if feature_type == 'raw':
		swp = (2 * wp) ** 2
		ndata = len(x_v)
		dwp = 2 * wp
		ndepths = len(depths)

		DATASET = np.zeros((ndata, swp * ndepths))
		images = {}
		for z in xrange(ndepths):
			images[z] = makesize(snd.zoom(IM, depths[z]), wp)

		X = [[int(x * depths[z]) for x in x_v] for z in xrange(ndepths)]
		Y = [[int(y * depths[z]) for y in y_v] for z in xrange(ndepths)]
		cub = np.zeros((ndepths, dwp, dwp))

		for j in xrange(ndata):
			for z in xrange(ndepths):
				im = images[z]
				cub[z, :, :] = im[Y[z][j]:Y[z][j] + dwp, X[z][j]:X[z][j] + dwp]
			DATASET[j, :] = cub.flatten()
		return DATASET
	elif feature_type == 'sub':
		swp = (2 * wp) ** 2
		ndata = len(x_v)
		dwp = 2 * wp
		ndepths = len(depths)
		DATASET = np.zeros((ndata, swp * ndepths))

		images = {}
		for z in xrange(ndepths):
			images[z] = makesize(snd.zoom(IM, depths[z]), wp)
		X = [[int(x * depths[z]) for x in x_v] for z in xrange(ndepths)]
		Y = [[int(y * depths[z]) for y in y_v] for z in xrange(ndepths)]
		cub = np.zeros((ndepths, dwp, dwp))
		for j in xrange(ndata):
			x = int(x_v[j])
			y = int(y_v[j])
			for z in xrange(ndepths):
				im = images[z]
				cub[z, :, :] = im[Y[z][j]:Y[z][j] + dwp, X[z][j]:X[z][j] + dwp] - IM[y, x]
			DATASET[j, :] = cub.flatten()
		return DATASET
	elif feature_type == 'haar':
		(coords_h2, coords_v2, coords_h3, coords_v3, coords_sq) = feature_parameters
		xo = np.array(x_v)
		yo = np.array(y_v)
		n_h2 = coords_h2.shape[0]
		n_v2 = coords_v2.shape[0]
		n_h3 = coords_h3.shape[0]
		n_v3 = coords_v3.shape[0]
		n_sq = coords_sq.shape[0]

		ndata = xo.size
		coords = np.zeros((ndata, 4))
		dataset = np.zeros((ndata, (n_h2 + n_v2 + n_h3 + n_v3 + n_sq) * depths.size))

		feature_index = 0

		for resolution in depths:

			if resolution == 1:
				rimg = IM
			else:
				rimg = snd.zoom(IM, resolution)

			intg = build_integral_slice(rimg)

			pad_intg = makesize(intg, 1)

			x = np.round((xo * resolution) + 1).astype(int)
			y = np.round((yo * resolution) + 1).astype(int)
			(h, w) = pad_intg.shape
			h -= 1
			w -= 1

			for i in range(n_h2):
				coords[:, 0] = (x + coords_h2[i, 0])
				coords[:, 1] = (y + coords_h2[i, 1])
				coords[:, 2] = (x + coords_h2[i, 2])
				coords[:, 3] = (y + coords_h2[i, 3])
				(xc, yc) = generate_2d_coordinates_horizontal(coords)
				xc = xc.clip(min=0, max=w)
				yc = yc.clip(min=0, max=h)
				zero = pad_intg[yc[:, 0], xc[:, 0]]
				un = pad_intg[yc[:, 1], xc[:, 1]]
				deux = pad_intg[yc[:, 2], xc[:, 2]]
				trois = pad_intg[yc[:, 3], xc[:, 3]]
				quatre = pad_intg[yc[:, 4], xc[:, 4]]
				cinq = pad_intg[yc[:, 5], xc[:, 5]]
				dataset[:, feature_index] = zero + (2 * un) + (-deux) + trois + (-2 * quatre) + cinq
				feature_index += 1

			for i in range(n_v2):
				coords[:, 0] = x + coords_v2[i, 0]
				coords[:, 1] = y + coords_v2[i, 1]
				coords[:, 2] = x + coords_v2[i, 2]
				coords[:, 3] = y + coords_v2[i, 3]
				(xc, yc) = generate_2d_coordinates_vertical(coords)
				xc = xc.clip(min=0, max=w)
				yc = yc.clip(min=0, max=h)
				zero = pad_intg[yc[:, 0], xc[:, 0]]
				un = pad_intg[yc[:, 1], xc[:, 1]]
				deux = pad_intg[yc[:, 2], xc[:, 2]]
				trois = pad_intg[yc[:, 3], xc[:, 3]]
				quatre = pad_intg[yc[:, 4], xc[:, 4]]
				cinq = pad_intg[yc[:, 5], xc[:, 5]]
				dataset[:, feature_index] = zero + (-un) + (-2 * deux) + (2 * trois) + quatre - cinq
				feature_index += 1

			for i in range(n_h3):
				coords[:, 0] = x + coords_h3[i, 0]
				coords[:, 1] = y + coords_h3[i, 1]
				coords[:, 2] = x + coords_h3[i, 2]
				coords[:, 3] = y + coords_h3[i, 3]
				(xc, yc) = generate_3d_coordinates_horizontal(coords)
				xc = xc.clip(min=0, max=w)
				yc = yc.clip(min=0, max=h)
				zero = pad_intg[yc[:, 0], xc[:, 0]]
				un = pad_intg[yc[:, 1], xc[:, 1]]
				deux = pad_intg[yc[:, 2], xc[:, 2]]
				trois = pad_intg[yc[:, 3], xc[:, 3]]
				quatre = pad_intg[yc[:, 4], xc[:, 4]]
				cinq = pad_intg[yc[:, 5], xc[:, 5]]
				six = pad_intg[yc[:, 6], xc[:, 6]]
				sept = pad_intg[yc[:, 7], xc[:, 7]]
				dataset[:, feature_index] = zero + (-2 * un) + (2 * deux) + (-trois) + (-quatre) + (2 * cinq) + (-2 * six) + sept
				feature_index += 1

			for i in range(n_v3):
				coords[:, 0] = x + coords_v3[i, 0]
				coords[:, 1] = y + coords_v3[i, 1]
				coords[:, 2] = x + coords_v3[i, 2]
				coords[:, 3] = y + coords_v3[i, 3]
				(xc, yc) = generate_3d_coordinates_vertical(coords)
				xc = xc.clip(min=0, max=w)
				yc = yc.clip(min=0, max=h)
				zero = pad_intg[yc[:, 0], xc[:, 0]]
				un = pad_intg[yc[:, 1], xc[:, 1]]
				deux = pad_intg[yc[:, 2], xc[:, 2]]
				trois = pad_intg[yc[:, 3], xc[:, 3]]
				quatre = pad_intg[yc[:, 4], xc[:, 4]]
				cinq = pad_intg[yc[:, 5], xc[:, 5]]
				six = pad_intg[yc[:, 6], xc[:, 6]]
				sept = pad_intg[yc[:, 7], xc[:, 7]]
				dataset[:, feature_index] = zero + (-un) + (-2 * deux) + (2 * trois) + (2 * quatre) + (-2 * cinq) + (-six) + sept
				feature_index += 1

			for i in range(n_sq):
				coords[:, 0] = x + coords_sq[i, 0]
				coords[:, 1] = y + coords_sq[i, 1]
				coords[:, 2] = x + coords_sq[i, 2]
				coords[:, 3] = y + coords_sq[i, 3]
				(xc, yc) = generate_square_coordinates(coords)
				xc = xc.clip(min=0, max=w)
				yc = yc.clip(min=0, max=h)
				zero = pad_intg[yc[:, 0], xc[:, 0]]
				un = pad_intg[yc[:, 1], xc[:, 1]]
				deux = pad_intg[yc[:, 2], xc[:, 2]]
				trois = pad_intg[yc[:, 3], xc[:, 3]]
				quatre = pad_intg[yc[:, 4], xc[:, 4]]
				cinq = pad_intg[yc[:, 5], xc[:, 5]]
				six = pad_intg[yc[:, 6], xc[:, 6]]
				sept = pad_intg[yc[:, 7], xc[:, 7]]
				huit = pad_intg[yc[:, 8], xc[:, 8]]
				dataset[:, feature_index] = zero + (-2 * un) + deux + (-2 * trois) + (4 * quatre) + (-2 * cinq) + six + (-2 * sept) + huit
				feature_index += 1
		return dataset
	elif feature_type == 'gaussian':
		xo = np.array(x_v)
		yo = np.array(y_v)
		feature_offsets = feature_parameters
		dataset = np.zeros((xo.size, feature_offsets[:, 0].size * depths.size))
		j = 0
		for kl in range(depths.size):
			resolution = depths[kl]
			rimg = snd.zoom(IM, resolution)
			rimg = makesize(rimg, 1)
			x = np.round((xo * resolution) + 1).astype(int)
			y = np.round((yo * resolution) + 1).astype(int)
			(h, w) = rimg.shape
			original_values = rimg[y, x]
			for i in range(feature_offsets[:, 0].size):
				dataset[:, j] = original_values - rimg[(y + feature_offsets[i, 1]).clip(min=0, max=h - 1), (x + feature_offsets[i, 0]).clip(min=0, max=w - 1)]
				j += 1
		return dataset
	return None


def rotate_coordinates(repository, num, x, y, angle, image_type):
	image = readimage(repository, num, image_type)
	if angle != 0:
		image_rotee = snd.rotate(image, -angle)
		(h, w) = image.shape
		(hr, wr) = image_rotee.shape
		angle_rad = angle * (np.pi / 180.)
		c = np.cos(angle_rad)
		s = np.sin(angle_rad)
		x -= (w / 2.)
		y -= (h / 2.)
		xr = ((x * c) - (y * s)) + (wr / 2.)
		yr = ((x * s) + (y * c)) + (hr / 2.)
		return xr.tolist(), yr.tolist(), image_rotee
	else:
		return x.tolist(), y.tolist(), image


def dataset_image_rot(repository, Xc, Yc, R, RMAX, proportion, step, i, ang, feature_type, feature_parameters, depths, window_size, image_type):
	x_v = []
	y_v = []
	REP = []
	IMGS = []
	deuxpi = 2. * np.pi
	for x in np.arange(np.int(Xc) - R, np.int(Xc) + R + 1, step):
		for y in np.arange(np.int(Yc) - R, np.int(Yc) + R + 1, step):
			if np.linalg.norm([Xc - x, Yc - y]) <= R:
				x_v.append(x)
				y_v.append(y)
				REP.append(1)
				IMGS.append(i)

	n = len(x_v)
	image = readimage(repository, i, image_type)
	(height, width) = image.shape
	height -= 1
	width -= 1
	for t in range(int(round(proportion * n))):
		angle = np.random.ranf() * deuxpi
		r = R + (np.random.ranf() * (RMAX - R))
		tx = int(r * np.cos(angle))
		ty = int(r * np.sin(angle))
		x_v.append(min(width, Xc + tx))
		y_v.append(min(height, Yc + ty))
		REP.append(0)
		IMGS.append(i)

	(x_r, y_r, im) = rotate_coordinates(repository, i, np.round(np.array(x_v)), np.round(np.array(y_v)), ang, image_type)
	(hr, wr) = im.shape
	hr -= 1
	wr -= 1

	x_r = np.round(x_r)
	y_r = np.round(y_r)

	for index in range(len(x_r)):
		xr = x_r[index]
		yr = y_r[index]
		if xr < 0:
			x_r[index] = 0
		if yr < 0:
			y_r[index] = 0
		if xr > wr:
			x_r[index] = wr
		if yr > hr:
			y_r[index] = hr

	return build_dataset_image(im, window_size, x_r, y_r, feature_type, feature_parameters, depths), REP, IMGS


def mp_helper_rot(job_args):
	return dataset_image_rot(*job_args)


def build_datasets_rot_mp(repository, imc, Xc, Yc, R, RMAX, proportion, step, ang, window_size, feature_type,  feature_parameters, depths, nimages, image_type, njobs):
	job_pool = Pool(njobs)
	job_args = [(repository, Xc[i], Yc[i], R, RMAX, proportion, step, imc[i], (np.random.ranf() * 2 * ang) - ang, feature_type, feature_parameters, depths, window_size, image_type) for i in range(nimages)]
	T = job_pool.map(mp_helper_rot, job_args)
	job_pool.close()
	job_pool.join()
	return T


def main():

	opt_parser = optparse.OptionParser(description='Cytomine Landmark Detection : Model building', prog='Cytomine Landmark Detector : Model builder', version='0.1')
	opt_parser.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	opt_parser.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key", help="Cytomine public key")
	opt_parser.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key", help="Cytomine private key")
	opt_parser.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	opt_parser.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	opt_parser.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	opt_parser.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	opt_parser.add_option('--cytomine_id_terms', type='string', dest='cytomine_id_terms', help="The identifiers of the terms to create detection models for. Terms must be separated by commas (no spaces). If 'all' is mentioned instead, every terms will be detected.")
	opt_parser.add_option('--cytomine_training_images', default='all', type='string', dest='cytomine_training_images', help="identifiers of the images used to create the models. ids must be separated by commas (no spaces). If 'all' is mentioned instead, every image with manual annotation will be used.")
	opt_parser.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	opt_parser.add_option('--model_njobs', type='int', default=1, dest='model_njobs', help="The number of processors used for model building")
	opt_parser.add_option('--model_R', type='int', default=6, dest='model_R', help="Max distance for extracting landmarks")
	opt_parser.add_option('--model_RMAX', type='int', default=200, dest='model_RMAX', help="Max distance for extracting non-landmarks")
	opt_parser.add_option('--model_P', type='float', default=3, dest='model_P', help="Proportion of non-landmarks")
	opt_parser.add_option('--model_npred', type='int', default=50000, dest='model_npred', help="Number of pixels extracted for prediction")
	opt_parser.add_option('--model_ntrees', type='int', default=50, dest='model_ntrees', help="Number of trees")
	opt_parser.add_option('--model_ntimes', type='int', default=3, dest='model_ntimes', help="Number of rotations to apply to the image")
	opt_parser.add_option('--model_angle', type='float', default=30, dest='model_angle', help="Max angle for rotation")
	opt_parser.add_option('--model_depth', type='int', default=5, dest='model_depth', help="Number of resolutions to use")
	opt_parser.add_option('--model_step', type='int', default=1, dest='model_step', help="Landmark pixels will be extracted in a grid (x-R:step:x+r,y-R:step:y+R) around the landmark")
	opt_parser.add_option('--model_wsize', type='int', default=8, dest='model_wsize', help="Window size")
	opt_parser.add_option('--model_feature_type', type='string', default='haar', dest='model_feature_type', help='The type of feature (raw, sub, haar or gaussian).')
	opt_parser.add_option('--model_feature_haar_n', type='int', default=1600, dest='model_feature_haar_n', help='Haar-Like features only. Number of descriptors for a pixel. Must be a multiple of 5*depths.')
	opt_parser.add_option('--model_feature_gaussian_n', type='int', default=1600, dest='model_feature_gaussian_n', help='Gaussian features only. Number of descriptors for a pixel. Must be a multiple of depths.')
	opt_parser.add_option('--model_feature_gaussian_std', type='float', default=20., dest='model_feature_gaussian_std', help='Gaussian features only. Standard deviation for the gaussian.')
	opt_parser.add_option('--model_save_to', type='string', default='/tmp/', dest='model_save_to', help="Destination for model storage")
	opt_parser.add_option('--model_name', type='string', dest='model_name', help="Name of the model (used for saving)")
	opt_parser.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")
	opts, argus = opt_parser.parse_args(args=sys.argv)
	conn = cytomine.Cytomine(opts.cytomine_host, opts.cytomine_public_key, opts.cytomine_private_key, base_path=opts.cytomine_base_path, working_path=opts.cytomine_working_path, verbose=str2bool(opts.verbose))
	current_user = conn.get_current_user()
	
	if not current_user.algo:
		user_job = conn.add_user_job(opts.cytomine_id_software, opts.cytomine_id_project)
		conn.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
		user_job = current_user

	job = conn.get_job(user_job.job)
	job = conn.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Bulding model...")
	job_parameters = {'cytomine_id_terms': opts.cytomine_id_terms, 'cytomine_training_images': opts.cytomine_training_images, 'model_R': opts.model_R, 'model_njobs': opts.model_njobs, 'model_RMAX': opts.model_RMAX, 'model_P': opts.model_P, 'model_npred': opts.model_npred, 'model_ntimes': opts.model_ntimes, 'model_angle': opts.model_angle, 'model_depth': opts.model_depth, 'model_wsize': opts.model_wsize, 'model_ntrees': opts.model_ntrees, 'model_step': opts.model_step, 'forest_max_features': ((2 * opts.model_wsize) ** 2) * opts.model_depth, 'forest_min_samples_split': 2, 'model_name': opts.model_name, 'model_feature_type': opts.model_feature_type, 'model_feature_haar_n': opts.model_feature_haar_n,
	'model_feature_gaussian_n': opts.model_feature_gaussian_n, 'model_feature_gaussian_std': opts.model_feature_gaussian_std}

	conn.add_job_parameters(user_job.job, conn.get_software(opts.cytomine_id_software), job_parameters)

	download_images(conn, opts.cytomine_id_project)
	download_annotations(conn, opts.cytomine_id_project, opts.cytomine_working_path)

	repository = opts.cytomine_working_path + str(opts.cytomine_id_project) + '/'
	txt_repository = opts.cytomine_working_path + '%d/txt/' % opts.cytomine_id_project

	depths = 1. / (2. ** np.arange(opts.model_depth))

	(xc, yc, xr, yr, ims, t_to_i, i_to_t) = getallcoords(txt_repository)

	if opts.cytomine_id_terms == 'all':
		term_list = t_to_i.keys()
	else:
		term_list = [int(term) for term in opts.cytomine_id_terms.split(',')]

	if opts.cytomine_training_images == 'all':
		tr_im = ims
	else:
		tr_im = [int(id_im) for id_im in opts.cytomine_training_images.split(',')]

	DATA = None
	REP = None
	be = 0
	model_repo = opts.model_save_to

	for id_term in term_list:
		(xc, yc, xr, yr) = getcoordsim(txt_repository, id_term, tr_im)

		nimages = np.max(xc.shape)
		mx = np.mean(xr)
		my = np.mean(yr)
		P = np.zeros((2, nimages))
		P[0, :] = xr
		P[1, :] = yr
		cm = np.cov(P)

		passe = False

		progress = 0
		delta = 80 / opts.model_ntimes

		# additional parameters
		feature_parameters = None
		if opts.model_feature_type.lower() == 'gaussian':
			std_matrix = np.eye(2) * (opts.model_feature_gaussian_std ** 2)
			feature_parameters = np.round(np.random.multivariate_normal([0, 0], std_matrix, opts.model_feature_gaussian_n)).astype(int)
		elif opts.model_feature_type.lower() == 'haar':
			W = opts.model_wsize
			n = opts.model_feature_haar_n / (5 * opts.model_depth)
			h2 = generate_2_horizontal(W, n)
			v2 = generate_2_vertical(W, n)
			h3 = generate_3_horizontal(W, n)
			v3 = generate_3_vertical(W, n)
			sq = generate_square(W, n)
			feature_parameters = (h2, v2, h3, v3, sq)

		for times in range(opts.model_ntimes):
			if times == 0:
				rangrange = 0
			else:
				rangrange = opts.model_angle

			T = build_datasets_rot_mp(repository, tr_im, xc, yc, opts.model_R, opts.model_RMAX, opts.model_P, opts.model_step, rangrange, opts.model_wsize, opts.model_feature_type, feature_parameters, depths, nimages, opts.image_type, opts.model_njobs)
			for i in range(len(T)):
				(data, rep, img) = T[i]
				(height, width) = data.shape
				if not passe:
					passe = True
					DATA = np.zeros((height * (len(T) + 100) * opts.model_ntimes, width))
					REP = np.zeros(height * (len(T) + 100) * opts.model_ntimes)
					b = 0
					be = height
				DATA[b:be, :] = data
				REP[b:be] = rep
				b = be
				be = be + height

			progress += delta
			job = conn.update_job_status(job, status=job.RUNNING, status_comment="Bulding model...", progress=progress)

		REP = REP[0:b]
		DATA = DATA[0:b, :]

		clf = ExtraTreesClassifier(n_jobs=opts.model_njobs, n_estimators=opts.model_ntrees)
		clf = clf.fit(DATA, REP)

		job = conn.update_job_status(job, status=job.RUNNING, progress=90, status_comment="Writing model...")

		if not os.path.isdir(model_repo):
			os.mkdir(model_repo)

		joblib.dump(clf, '%s%s_%d.pkl' % (model_repo, opts.model_name, id_term))
		joblib.dump([mx, my, cm], '%s%s_%d_cov.pkl' % (model_repo, opts.model_name, id_term))
		if opts.model_feature_type == 'haar' or opts.model_feature_type == 'gaussian':
			joblib.dump(feature_parameters, '%s%s_%d_fparameters.pkl' % (model_repo, opts.model_name, id_term))

	f = open('%s%s.conf' % (model_repo, opts.model_name), 'w+')
	f.write('cytomine_id_terms %s\n' % opts.cytomine_id_terms)
	f.write('model_R %d\n' % opts.model_R)
	f.write('model_RMAX %d\n' % opts.model_RMAX)
	f.write('model_P %f\n' % opts.model_P)
	f.write('model_npred %d\n' % opts.model_npred)
	f.write('model_ntrees %d\n' % opts.model_ntrees)
	f.write('model_ntimes %d\n' % opts.model_ntimes)
	f.write('model_angle %f\n' % opts.model_angle)
	f.write('model_depth %d\n' % opts.model_depth)
	f.write('model_step %d\n' % opts.model_step)
	f.write('window_size %d\n' % opts.model_wsize)
	f.write('feature_type %s\n' % opts.model_feature_type)
	f.write('feature_haar_n %d\n' % opts.model_feature_haar_n)
	f.write('feature_gaussian_n %d\n' % opts.model_feature_gaussian_n)
	f.write('feature_gaussian_std %f' % opts.model_feature_gaussian_std)
	f.close()
	conn.update_job_status(job, status=job.TERMINATED, progress=100, status_comment="Model built!")

if __name__ == "__main__":
	main()
