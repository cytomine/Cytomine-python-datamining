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

from sklearn.externals import joblib
from download import *
import optparse
import cytomine
from VotingTreeRegressor import *
from build_integral_image import *
from ldmtools import *
import sys

def pad_integral(intg):
	(h, w) = intg.shape
	nintg = np.zeros((h+1, w+1))
	nintg[1:, 1:] = intg
	return nintg


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


def compute_features(intg, x, y, coords_h2, coords_v2, coords_h3, coords_v3, coords_sq):
	pad_intg = pad_integral(intg)
	x += 1
	y += 1
	(h, w) = pad_intg.shape
	h -= 1
	w -= 1

	(n_h2, quatre) = coords_h2.shape
	(n_v2, quatre) = coords_v2.shape
	(n_h3, quatre) = coords_h3.shape
	(n_v3, quatre) = coords_v3.shape
	(n_sq, quatre) = coords_sq.shape

	ndata = x.size
	coords = np.zeros((ndata, 4))
	dataset = np.zeros((ndata, n_h2 + n_v2 + n_h3 + n_v3 + n_sq))
	feature_index = 0

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


def build_dataset_image_offset(repository, image_number, xc, yc, dmax, nsamples, h2, v2, h3, v3, sq):
	intg = build_integral_image(readimage(repository, image_number))
	rep_x = np.random.randint(-dmax, dmax + 1, nsamples)
	rep_y = np.random.randint(-dmax, dmax + 1, nsamples)
	xs = xc + rep_x
	ys = yc + rep_y
	rep = np.zeros((nsamples, 2))
	rep[:, 0] = rep_x
	rep[:, 1] = rep_y
	return compute_features(intg, xs, ys, h2, v2, h3, v3, sq), rep


def bdio_helper(jobargs):
	return build_dataset_image_offset(*jobargs)


def build_dataset_image_offset_mp(repository, xc, yc, ims, dmax, nsamples, h2, v2, h3, v3, sq, n_jobs):
	nimages = xc.size
	jobargs = [(repository, ims[image_number], xc[image_number], yc[image_number], dmax, nsamples, h2, v2, h3, v3, sq) for image_number in range(nimages)]
	P = Pool(n_jobs)
	T = P.map(bdio_helper, jobargs)
	P.close()
	P.join()
	DATASET = None
	REP = None
	IMG = None

	b = 0

	for i in range(nimages):
		(data, r) = T[i]
		if i == 0:
			(h, w) = data.shape
			DATASET = np.zeros((h * nimages, w))
			REP = np.zeros((h * nimages, 2))
			IMG = np.zeros(h * nimages)
		next_b = b + h
		DATASET[b:next_b, :] = data
		REP[b:next_b, :] = r
		IMG[b:next_b] = i
		b = next_b

	return DATASET, REP, IMG


def procrustes(x_coords, y_coords):
	(ndata, nldms) = x_coords.shape

	mean_x = np.mean(x_coords, axis=1)
	mean_y = np.mean(y_coords, axis=1)
	coords_centered = np.zeros((ndata, 2 * nldms))
	for i in range(nldms):
		coords_centered[:, i] = x_coords[:, i] - mean_x
		coords_centered[:, i + nldms] = y_coords[:, i] - mean_y

	coords_centered[0, :] = coords_centered[0, :] / np.linalg.norm(coords_centered[0, :])

	c = np.zeros((2, nldms))
	for i in range(1, ndata):
		a = np.dot(coords_centered[i, :], coords_centered[0, :]) / (np.linalg.norm(coords_centered[i, :]) ** 2)
		b = np.sum((coords_centered[i, 0:nldms] * coords_centered[0, nldms:]) - (coords_centered[0, :nldms] * coords_centered[i, nldms:])) / (np.linalg.norm(coords_centered[i, :]) ** 2)
		s = np.sqrt((a ** 2) + (b ** 2))
		theta = np.arctan(b / a)

		scaling_matrix = s * np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
		c[0, :] = coords_centered[i, 0:nldms]
		c[1, :] = coords_centered[i, nldms:]

		new_c = np.dot(scaling_matrix, c)

		coords_centered[i, 0:nldms] = new_c[0, :]
		coords_centered[i, nldms:] = new_c[1, :]

	return coords_centered


def apply_pca(coords, k):
	(ndata, nldms) = coords.shape
	m = np.mean(coords, axis=0).reshape((nldms, 1))
	mat = np.zeros((nldms, nldms))
	for i in range(ndata):
		v = coords[i, :].reshape((nldms, 1))
		d = v - m
		mat += np.dot(d, d.T)
	mat /= float(nldms - 1)
	(values, vectors) = np.linalg.eig(mat)
	return m, vectors[:, 0:k]


def main():
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Model building', prog='Cytomine Landmark Detector : Model builder', version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--cytomine_training_images', type='string', default='all', dest='cytomine_training_images', help="IDs of the training images. Must be separated by commas, no spaces. 'all' takes all the available annotated images.")
	p.add_option('--cytomine_id_terms', type='string', default=1, dest='cytomine_id_terms', help="The identifiers of the terms to create detection models for. Terms must be separated by commas (no spaces). If 'all' is mentioned instead, every terms will be detected.")
	p.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--model_njobs', type='int', default=4, dest='model_njobs', help="The number of processors used for model building")
	p.add_option('--model_D_MAX', type='int', default=6, dest='model_D_MAX', help="D_MAX parameter.")
	p.add_option('--model_n_samples', type='int', default=200, dest='model_n_samples', help="Number of samples for phase 1.")
	p.add_option('--model_W', type='int', default=100, dest='model_W', help="Window size for feature extraction.")
	p.add_option('--model_n', type='int', default=20, dest='model_n', help="Number of samples extracted.")
	p.add_option('--model_T', type='int', default=50, dest='model_T', help="Number of trees for phase 1.")
	p.add_option('--model_step', type='int', default=3, dest='model_step', help="Step for prediction for phase 1.")
	p.add_option('--model_n_reduc', type='int', default=2, dest='model_n_reduc', help="Size for PCA reduction in phase 2.")
	p.add_option('--model_R_MAX', type='int', default=20, dest='model_R_MAX', help="Maximal radius for phase 2.")
	p.add_option('--model_R_MIN', type='int', default=3, dest='model_R_MIN', help="Minimal radius for phase 2.")
	p.add_option('--model_alpha', type='float', default=0.5, dest='model_alpha', help="Radius reduction parameter for phase 2.")
	p.add_option('--model_save_to', type='string', default='/tmp/', dest='model_save_to', help="Destination for model storage")
	p.add_option('--model_name', type='string', dest='model_name', help="Name of the model (used for saving)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	options, arguments = p.parse_args(args=sys.argv)
	options.cytomine_working_path = options.cytomine_working_path.rstrip('/') + '/'

	cytomine_connection = cytomine.Cytomine(options.cytomine_host, options.cytomine_public_key, options.cytomine_private_key, base_path=options.cytomine_base_path, working_path=options.cytomine_working_path, verbose=options.verbose)

	current_user = cytomine_connection.get_current_user()
	if not current_user.algo:
		user_job = cytomine_connection.add_user_job(options.cytomine_id_software, options.cytomine_id_project)
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
		user_job = current_user

	job = cytomine_connection.get_job(user_job.job)
	cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Bulding model...")
	model_repo = options.model_save_to

	if not os.path.isdir(model_repo):
		os.mkdir(model_repo)

	download_images(cytomine_connection, options.cytomine_id_project)
	download_annotations(cytomine_connection, options.cytomine_id_project, options.cytomine_working_path)

	repository = options.cytomine_working_path + str(options.cytomine_id_project) + '/'
	(xc, yc, xr, yr, ims, term_to_i, i_to_term) = getallcoords(repository.rstrip('/') + '/txt/')
	(nims, nldms) = xc.shape

	if options.cytomine_id_terms != 'all':
		term_list = [int(term) for term in options.cytomine_id_terms.split(',')]
		Xc = np.zeros((nims, len(term_list)))
		Yc = np.zeros(Xc.shape)
		i = 0
		for id_term in term_list:
			Xc[:, i] = xc[:, term_to_i[id_term]]
			Yc[:, i] = yc[:, term_to_i[id_term]]
			i += 1
	else:
		term_list = term_to_i.keys()
		Xc = xc
		Yc = yc
	(nims, nldms) = Xc.shape

	if options.cytomine_training_images != 'all':
		im_list = [int(p) for p in options.cytomine_training_images.split(',')]
	else:
		im_list = ims

	X = np.zeros((len(im_list), nldms))
	Y = np.zeros((len(im_list), nldms))
	im_to_i = {}

	for i in range(nims):
		im_to_i[ims[i]] = i
	for i in range(len(im_list)):
		X[i, :] = Xc[im_to_i[im_list[i]], :]
		Y[i, :] = Yc[im_to_i[im_list[i]], :]

	Xc = X
	Yc = Y
	h2 = generate_2_horizontal(options.model_W, options.model_n)
	v2 = generate_2_vertical(options.model_W, options.model_n)
	h3 = generate_3_horizontal(options.model_W, options.model_n)
	v3 = generate_3_vertical(options.model_W, options.model_n)
	sq = generate_square(options.model_W, options.model_n)

	joblib.dump((h2, v2, h3, v3, sq), '%s%s_lc_feature_map.pkl' % (model_repo, options.model_name))
	for id_term in term_list:
		(dataset, rep, img) = build_dataset_image_offset_mp(repository, Xc[:, term_to_i[id_term]], Yc[:, term_to_i[id_term]], im_list, options.model_D_MAX, options.model_n_samples, h2, v2, h3, v3, sq, options.model_njobs)
		clf = VotingTreeRegressor(n_estimators=options.model_T, n_jobs=options.model_njobs)
		clf = clf.fit(dataset, rep)
		joblib.dump(clf, '%s%s_lc_regressor_%d.pkl' % (model_repo, options.model_name, id_term))
	xt = procrustes(Xc, Yc)
	(mu, P) = apply_pca(xt, options.model_n_reduc)
	joblib.dump((mu, P, np.mean(Xc, axis=0), np.mean(Yc, axis=0)), '%s%s_lc_pca.pkl' % (model_repo, options.model_name))
	F = open('%s%s_lc_parameters.conf' % (options.model_save_to, options.model_name), 'wb')
	F.write('cytomine_id_terms %s\n' % options.cytomine_id_terms)
	F.write('model_njobs %d\n' % options.model_njobs)
	F.write('model_D_MAX %d\n' % options.model_D_MAX)
	F.write('model_n_samples %d\n' % options.model_n_samples)
	F.write('model_W %d\n' % options.model_W)
	F.write('model_n %d\n' % options.model_n)
	F.write('model_T %d\n' % options.model_T)
	F.write('model_step %d\n' % options.model_step)
	F.write('model_n_reduc %d\n' % options.model_n_reduc)
	F.write('model_R_MAX %d\n' % options.model_R_MAX)
	F.write('model_R_MIN %d\n' % options.model_R_MIN)
	F.write('model_alpha %f\n' % options.model_alpha)
	F.close()

if __name__ == "__main__":
	main()
