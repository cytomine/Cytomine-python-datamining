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
import scipy.ndimage as snd

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