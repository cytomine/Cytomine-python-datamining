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

from ldmtools import *
import optparse
import sys
from sklearn.externals import joblib
from shapely.geometry import Point
from cytomine import cytomine, models
from build_integral_image import *

"""
Phase 1 : Pixel filtering
"""


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


def pad_integral(intg):
	(h, w) = intg.shape
	nintg = np.zeros((h + 1, w + 1))
	nintg[1:, 1:] = intg
	return nintg


def compute_features(intg, x, y, coords_h2, coords_v2, coords_h3, coords_v3, coords_sq):
	pad_intg = pad_integral(intg)
	x = x + 1
	y = y + 1
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
		dataset[:, feature_index] = zero + (-2 * un) + (2 * deux) + (-trois) + (-quatre) + (2 * cinq) + (
			-2 * six) + sept
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
		dataset[:, feature_index] = zero + (-un) + (-2 * deux) + (2 * trois) + (2 * quatre) + (-2 * cinq) + (
			-six) + sept
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
		dataset[:, feature_index] = zero + (-2 * un) + deux + (-2 * trois) + (4 * quatre) + (-2 * cinq) + six + (
			-2 * sept) + huit
		feature_index += 1

	return dataset


def build_vote_map(repository, image_number, clf, h2, v2, h3, v3, sq, stepc):
	intg = build_integral_image(readimage(repository, image_number))
	(h, w) = intg.shape

	vote_map = np.zeros((h, w))

	coords = np.array([[x, y] for x in range(0, w, stepc) for y in range(0, h, stepc)]).astype(int)

	y_v = coords[:, 1]
	x_v = coords[:, 0]

	step = 50000

	b = 0

	rep = np.zeros((step, 2))

	hash_map = {}

	while (b < x_v.size):
		b_next = min(b + step, x_v.size)
		offsets = clf.predict(compute_features(intg, x_v[b:b_next], y_v[b:b_next], h2, v2, h3, v3, sq))
		n_trees = len(offsets)
		off_size = int(b_next - b)

		offsets = np.array(offsets)
		toffsize = off_size * n_trees
		offsets = offsets.reshape((toffsize, 2))

		offsets[:, 0] = np.tile(x_v[b:b_next], n_trees) - offsets[:, 0]
		offsets[:, 1] = np.tile(y_v[b:b_next], n_trees) - offsets[:, 1]

		t, = np.where(offsets[:, 0] < 0)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 1] < 0)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 0] >= w)
		offsets = np.delete(offsets, t, axis=0)
		t, = np.where(offsets[:, 1] >= h)
		offsets = np.delete(offsets, t, axis=0)
		(toffsize, tamere) = offsets.shape
		for i in range(toffsize):
			vote_map[int(offsets[i, 1]), int(offsets[i, 0])] += 1

		b = b_next

	return vote_map


def find_best_positions(vote_map, coords, R):
	(h, w, nldms) = vote_map.shape

	cs = np.zeros(2 * nldms)
	for ip in range(nldms):

		x_begin = min(w - 1, max(0, coords[ip] - R))
		x_end = max(0, min(coords[ip] + R + 1, w - 1))

		y_begin = min(h - 1, max(0, coords[ip + nldms] - R))
		y_end = max(0, min(h - 1, coords[ip + nldms] + R + 1))

		if (x_begin != x_end and y_begin != y_end):
			window = vote_map[y_begin:y_end, x_begin:x_end, ip]
			(y, x) = np.where(window == np.max(window))
			cs[ip] = x[0] + x_begin
			cs[ip + nldms] = y[0] + y_begin
		elif (x_begin == x_end and y_begin != y_end):
			window = vote_map[y_begin:y_end, x_begin, ip]
			y, = np.where(window == np.max(window))
			cs[ip] = x_begin
			cs[ip + nldms] = y[0] + y_begin
		elif (y_begin == y_end and x_begin != x_end):
			window = vote_map[y_begin, x_begin:x_end, ip]
			x, = np.where(window == np.max(window))
			cs[ip + nldms] = y_begin
			cs[ip] = x[0] + x_begin
		else:
			cs[ip] = x_begin
			cs[ip + nldms] = y_begin

	return cs


def fit_shape(mu, P, ty):
	y = np.copy(ty)

	(nldms, k) = P.shape
	b = np.zeros((k, 1))
	nldm = nldms / 2
	c = np.zeros((2, nldm))
	new_y = np.zeros(nldms)

	m_1 = np.mean(y[:nldm])
	m_2 = np.mean(y[nldm:])

	y[:nldm] = y[:nldm] - m_1
	y[nldm:] = y[nldm:] - m_2

	ite = 0

	while (ite < 100):
		x = mu + np.dot(P, b)
		n2 = np.linalg.norm(y) ** 2
		a = (np.dot(y, x) / n2)[0]
		b = np.sum((y[:nldm] * x[nldm:]) - (x[:nldm] * y[nldm:])) / n2
		s = np.sqrt((a ** 2) + (b ** 2))
		theta = np.arctan(b / a)
		scaling_matrix = s * np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
		c[0, :] = y[:nldm]
		c[1, :] = y[nldm:]

		# sys.exit()
		new_c = np.dot(scaling_matrix, c)

		new_y[:nldm] = new_c[0, :]
		new_y[nldm:] = new_c[1, :]

		b = np.dot(P.T, new_y.reshape((nldms, 1)) - mu)
		# y = new_y
		ite += 1

	s = 1. / s
	theta = -theta
	scaling_matrix = s * np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
	c[0, :] = x[:nldm].reshape(nldm)
	c[1, :] = x[nldm:].reshape(nldm)
	new_c = np.dot(scaling_matrix, c)
	new_y[:nldm] = new_c[0, :] + m_1
	new_y[nldm:] = new_c[1, :] + m_2
	return new_y


if __name__ == "__main__":

	p = optparse.OptionParser(description='Cytomine Landmark Detection : Landmark Detection',
	                          prog='Cytomine Landmark Detection : Landmark Dectector', version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host",
	             help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key",
	             help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key",
	             help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software",
	             help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path",
	             help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path",
	             help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project",
	             help="The Cytomine project identifier")
	p.add_option('--cytomine_predict_images', type='string', dest='cytomine_predict_images',
	             help='The identifier of the images to predict, separated by a comma (no spaces).')
	p.add_option('--model_load_from', default='/tmp/', type="string", dest="model_load_from",
	             help="The repository where the models are stored")

	p.add_option('--model_name', type="string", dest="model_name",
	             help="The name of the model to use for detection")

	p.add_option('--image_type', type='string', default='jpg', dest='image_type',
	             help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	options, arguments = p.parse_args(args=sys.argv)

	parameters = {
		'cytomine_host': options.cytomine_host,
		'cytomine_public_key': options.cytomine_public_key,
		'cytomine_private_key': options.cytomine_private_key,
		'cytomine_id_software': options.cytomine_id_software,
		'cytomine_base_path': options.cytomine_base_path,
		'cytomine_working_path': options.cytomine_working_path,
		'cytomine_id_project': options.cytomine_id_project,
		'cytomine_predict_images': options.cytomine_predict_images,
		'model_load_from': options.model_load_from.rstrip('/') + '/',
		'model_name': options.model_name,
		'image_type': options.image_type,
		'verbose': str2bool(options.verbose)
	}

	parameters['cytomine_working_path'] = parameters['cytomine_working_path'].rstrip('/') + '/'
	parameters['model_load_from'] = parameters['model_load_from'].rstrip('/') + '/'
	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'], parameters['cytomine_public_key'],
	                                        parameters['cytomine_private_key'],
	                                        base_path=parameters['cytomine_base_path'],
	                                        working_path=parameters['cytomine_working_path'],
	                                        verbose=parameters['verbose'])

	current_user = cytomine_connection.get_current_user()
	run_by_user_job = False
	id_software = parameters['cytomine_id_software']
	repository = options.cytomine_working_path + str(options.cytomine_id_project) + '/'

	if current_user.algo:
		sys.path.append('/software_router/algo/landmark_model_builder/')
		user_job = current_user
		run_by_user_job = True
	else:
		sys.path.append('../landmark_model_builder/')
		user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'],
		                                            parameters['cytomine_id_project'])
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))

	from download import *
	from ldmtools import *

	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0,
	                                            status_comment="Uploading annotations...")


	par = {}
	F = open('%s%s_lc_parameters.conf' % (parameters['model_load_from'], options.model_name))
	for l in F.readlines():
		tab = l.split()
		par[tab[0]] = tab[1]
	F.close()

	txt_repository = options.cytomine_working_path + '%d/txt/' % options.cytomine_id_project
	(xc, yc, xr, yr, ims, t_to_i, i_to_t) = getallcoords(txt_repository)
	if options.cytomine_predict_images == 'all':
		pr_im = ims
	else:
		pr_im = [int(id_im) for id_im in options.cytomine_predict_images.split(',')]
	if par['cytomine_id_terms'] == 'all':
		id_terms = np.sort(t_to_i.keys())
	else:
		id_terms = np.sort([int(x) for x in par['cytomine_id_terms'].split(',')])

	idim_to_i = {}
	for i in range(len(ims)):
		idim_to_i[ims[i]] = i

	model_repo = options.model_load_from

	(h2, v2, h3, v3, sq) = joblib.load('%s%s_lc_feature_map.pkl' % (model_repo, options.model_name))
	for id_term in id_terms:
		reg = joblib.load('%s%s_lc_regressor_%d.pkl' % (model_repo, options.model_name, id_term))
		for id_img in pr_im:
			vote_map = build_vote_map(repository, id_img, reg, h2, v2, h3, v3, sq, int(par['model_step']))
			np.savez_compressed(
				'%slc_votemap-%s-%d-%d.npy' % (repository, options.model_name, id_img, id_term), vote_map)
	(h, w) = vote_map.shape
	vote_map = None

	R_max = int(par['model_R_MAX'])
	R_min = int(par['model_R_MIN'])
	alpha = float(par['model_alpha'])
	(mu, P, mx, my) = joblib.load('%s%s_lc_pca.pkl' % (model_repo, options.model_name))
	hash_error = {}
	coords = np.zeros(id_terms.size * 2)
	for id_img in pr_im:
		global_vote_map = np.zeros((h, w, id_terms.size))
		for i in range(id_terms.size):
			global_vote_map[:, :, i] = \
				np.load('%slc_votemap-%s-%d-%d.npy.npz' % (repository, options.model_name, id_img, id_terms[i]))[
					'arr_0']

		current_R = R_max

		coords[:id_terms.size] = mx
		coords[id_terms.size:] = my

		while (current_R >= R_min):
			coords = np.round(find_best_positions(global_vote_map, coords, int(np.round(current_R)))).astype(int)
			coords = np.round(fit_shape(mu, P, coords)).astype(int)
			current_R = current_R * alpha

		for i in range(id_terms.size):
			x = coords[i]
			y = coords[i + id_terms.size]
			id_term = id_terms[i]
			circle = Point(x, y)
			location = circle.wkt
			new_annotation = cytomine_connection.add_annotation(location, id_img)
			cytomine_connection.add_annotation_term(new_annotation.id, term=id_term, expected_term=id_term, rate=1.0,
			                                        annotation_term_model=models.AlgoAnnotationTerm)

			xreal = xc[idim_to_i[id_img], t_to_i[id_term]]
			yreal = yc[idim_to_i[id_img], t_to_i[id_term]]
			er = np.linalg.norm([xreal - x, yreal - y])
			tup = (id_img, id_term)
			if (tup in hash_error and er < hash_error[tup]) or (not (tup in hash_error)):
				hash_error[(id_img, id_term)] = er

	job = cytomine_connection.update_job_status(job, status=job.TERMINATED, progress=100,
	                                            status_comment="Annotations uploaded!")

	csv_line = ";"

	for id_term in id_terms:
		csv_line += "%d;" % id_term
	csv_line += "img_avg;\n"

	cerror = {}
	for id_term in id_terms:
		cerror[id_term] = []

	for id_img in pr_im:
		csv_line += "%d;" % id_img
		lerror = []
		term_i = 0
		for id_term in id_terms:
			lerror.append(hash_error[(id_img, id_term)])
			csv_line += "%3.3f;" % hash_error[(id_img, id_term)]
			cerror[id_term].append(hash_error[(id_img, id_term)])
		csv_line += "%3.3f;\n" % np.mean(lerror)

	csv_line += "term_avg;"
	for id_term in id_terms:
		csv_line += "%3.3f;" % np.mean(cerror[id_term])
	csv_line += "%3.3f;" % np.mean(hash_error.values())
	job_parameters = {'cytomine_models': parameters['model_name'],
	                  'cytomine_predict_images': parameters['cytomine_predict_images'], 'prediction_error': csv_line}
	job_parameters_values = cytomine_connection.add_job_parameters(user_job.job,
	                                                               cytomine_connection.get_software(id_software),
	                                                               job_parameters)
