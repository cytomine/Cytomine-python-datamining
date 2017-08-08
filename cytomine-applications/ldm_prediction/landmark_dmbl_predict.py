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

from multiprocessing import Pool
import scipy.ndimage as snd
from scipy.stats import multivariate_normal
from sumproduct import Variable, Factor, FactorGraph
import optparse
import sys
from sklearn.externals import joblib
from shapely.geometry import Point
from cytomine import cytomine, models
from ldmtools import *


def dataset_from_coordinates(img, x, y, feature_offsets):
	(h, w) = img.shape
	original_values = img[y.clip(min=0, max=h - 1), x.clip(min=0, max=w - 1)]
	dataset = np.zeros((x.size, feature_offsets[:, 0].size))
	for i in range(feature_offsets[:, 0].size):
		dataset[:, i] = original_values - img[
			(y + feature_offsets[i, 1]).clip(min=0, max=h - 1), (x + feature_offsets[i, 0]).clip(min=0, max=w - 1)]
	return dataset


def image_dataset_phase_1(repository, image_number, x, y, feature_offsets, R_offsets, delta, P):
	img = makesize(snd.zoom(readimage(repository, image_number), delta), 1)
	(h, w) = img.shape
	mask = np.ones((h, w), 'bool')
	mask[:, 0] = 0
	mask[0, :] = 0
	mask[h - 1, :] = 0
	mask[:, w - 1] = 0
	(nroff, blc) = R_offsets.shape

	h -= 2
	w -= 2
	x += 1
	y += 1

	rep = np.zeros((x.size * nroff) + (P * nroff))
	xs = np.zeros((x.size * nroff) + (P * nroff)).astype('int')
	ys = np.zeros((x.size * nroff) + (P * nroff)).astype('int')
	for ip in range(x.size):
		xs[ip * nroff:(ip + 1) * nroff] = x[ip] + R_offsets[:, 0]
		ys[ip * nroff:(ip + 1) * nroff] = y[ip] + R_offsets[:, 1]
		rep[ip * nroff:(ip + 1) * nroff] = ip
	mask[ys, xs] = 0
	(ym, xm) = np.where(mask == 1)
	perm = np.random.permutation(ym.size)[0:P * nroff]
	ym = ym[perm]
	xm = xm[perm]
	xs[x.size * nroff:] = xm
	ys[y.size * nroff:] = ym
	rep[x.size * nroff:] = x.size

	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep


def dataset_mp_helper(jobargs):
	return image_dataset_phase_1(*jobargs)


def get_dataset_phase_1(repository, n_jobs, feature_offsets, R_offsets, delta, P):
	p = Pool(n_jobs)
	(Xc, Yc) = getcoords(repository.rstrip('/') + '/txt/')
	Xc = np.round(Xc * delta).astype('int')
	Yc = np.round(Yc * delta).astype('int')
	(nims, nldms) = Xc.shape
	jobargs = [(repository, i, Xc[i, :], Yc[i, :], feature_offsets, R_offsets, delta, P) for i in range(nims)]
	data = p.map(dataset_mp_helper, jobargs)
	p.close()
	p.join()

	(nroff, blc) = R_offsets.shape

	DATASET = np.zeros((nims * (nroff * (nldms + P)), feature_offsets[:, 0].size))
	REP = np.zeros(nims * (nroff * (nldms + P)))
	IMG = np.zeros(nims * (nroff * (nldms + P)))
	b = 0
	i = 0
	for (d, r) in data:
		(nd, nw) = d.shape
		DATASET[b:b + nd, :] = d
		REP[b:b + nd] = r
		IMG[b:b + nd] = i
		i += 1
		b = b + nd
	DATASET = DATASET[0:b, :]
	REP = REP[0:b]
	IMG = IMG[0:b]

	return DATASET, REP, IMG


def build_phase_1_model(repository, n_jobs=1, NT=32, F=100, R=2, sigma=10, delta=0.25, P=1):
	std_matrix = np.eye(2) * (sigma ** 2)
	feature_offsets = np.round(np.random.multivariate_normal([0, 0], std_matrix, NT * F)).astype('int')

	R_offsets = []
	for x1 in range(-R, R + 1):
		for x2 in range(-R, R + 1):
			if np.linalg.norm([x1, x2]) <= R:
				R_offsets.append([x1, x2])

	R_offsets = np.array(R_offsets).astype('int')

	(dataset, rep, img) = get_dataset_phase_1(repository, n_jobs, feature_offsets, R_offsets, delta, P)

	return dataset, rep, img, feature_offsets


def probability_map_phase_1(repository, image_number, clf, feature_offsets, delta):
	img = makesize(snd.zoom(readimage(repository, image_number), delta), 1)
	(h, w) = img.shape
	c = np.arange((h - 2) * (w - 2))
	ys = 1 + np.round(c / (w - 2)).astype('int')
	xs = 1 + np.mod(c, (w - 2))

	step = 20000
	b = 0
	probability_map = None
	nldms = -1

	while (b < xs.size):

		next_b = min(b + step, xs.size)
		dataset = dataset_from_coordinates(img, xs[b:next_b], ys[b:next_b], feature_offsets)
		probabilities = clf.predict_proba(dataset)

		if nldms == -1:
			(ns, nldms) = probabilities.shape
			probability_map = np.zeros((h - 2, w - 2, nldms))

		for ip in range(nldms):
			probability_map[ys[b:next_b] - 1, xs[b:next_b] - 1, ip] = probabilities[:, ip]
		b = next_b

	return probability_map


def image_dataset_phase_2(repository, image_number, x, y, feature_offsets, R_offsets, delta, P):
	img = makesize(snd.zoom(readimage(repository, image_number), delta), 1)
	(h, w) = img.shape
	mask = np.ones((h, w), 'bool')
	mask[:, 0] = 0
	mask[0, :] = 0
	mask[h - 1, :] = 0
	mask[:, w - 1] = 0
	(nroff, blc) = R_offsets.shape

	h -= 2
	w -= 2
	x += 1
	y += 1

	rep = np.zeros((x.size * nroff, 2))
	xs = np.zeros(x.size * nroff).astype('int')
	ys = np.zeros(x.size * nroff).astype('int')
	number = np.zeros(x.size * nroff)

	for ip in range(x.size):
		xs[ip * nroff:(ip + 1) * nroff] = x[ip] + R_offsets[:, 0]
		ys[ip * nroff:(ip + 1) * nroff] = y[ip] + R_offsets[:, 1]
		number[ip * nroff:(ip + 1) * nroff] = ip
		rep[ip * nroff:(ip + 1) * nroff, 0] = R_offsets[:, 0]
		rep[ip * nroff:(ip + 1) * nroff, 1] = R_offsets[:, 1]

	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep, number


def dataset_mp_helper_phase_2(jobargs):
	return image_dataset_phase_2(*jobargs)


def filter_perso(img, filter_size):
	offsets = []
	r = range(-filter_size, filter_size + 1)
	for r1 in r:
		for r2 in r:
			if (np.linalg.norm([r1, r2]) <= filter_size and r1 != 0 and r2 != 0):
				offsets.append([r1, r2])
	offsets = np.array(offsets)

	(h, w) = img.shape
	y, x = np.where(img > 0.)
	nimg = np.zeros((h, w))
	for i in range(x.size):
		val = img[y[i], x[i]]
		if np.sum(val < img[(y[i] + offsets[:, 1]).clip(min=0, max=h - 1), (x[i] + offsets[:, 0]).clip(min=0, max=w - 1)]) == 0:
			nimg[y[i], x[i]] = val

	return nimg


def agregation_phase_2(repository, image_number, ip, probability_maps, reg, delta, feature_offsets, filter_size, beta, n_iterations):
	img = makesize(snd.zoom(readimage(repository, image_number), delta), 1)
	(h, w, nldms) = probability_maps.shape
	nldms -= 1
	mh = h - 1
	mw = w - 1
	for iteration in range(n_iterations):
		y, x = np.where(probability_maps[:, :, ip] >= beta * np.max(probability_maps[:, :, ip]))
		dataset = dataset_from_coordinates(img, x + 1, y + 1, feature_offsets)
		offsets = reg.predict(dataset)
		n_x = (x - offsets[:, 0]).clip(min=0, max=mw)
		n_y = (y - offsets[:, 1]).clip(min=0, max=mh)
		new_pmap = np.zeros((h, w))
		for i in range(n_x.size):
			new_pmap[n_y[i], n_x[i]] += probability_maps[y[i], x[i], ip]
		probability_maps[:, :, ip] = new_pmap
		probability_maps[0, :, ip] = 0
		probability_maps[:, 0, ip] = 0
		probability_maps[mh, :, ip] = 0
		probability_maps[:, mw, ip] = 0

	return filter_perso(probability_maps[:, :, ip], filter_size)


def build_bmat_phase_3(xc, yc, T, x_candidates, y_candidates, edges, sde):
	B_mat = {}
	(nims, nldms) = xc.shape
	c1 = np.zeros((nims, 2))
	c2 = np.zeros((nims, 2))

	std_matrix = np.eye(2) * (sde ** 2)

	for ip in range(nldms):
		c1[:, 0] = xc[:, ip]
		c1[:, 1] = yc[:, ip]
		for ipl in edges[ip, :]:
			rel = np.zeros((len(x_candidates[ip]), len(x_candidates[ipl])))

			c2[:, 0] = xc[:, ipl]
			c2[:, 1] = yc[:, ipl]

			diff = c1 - c2

			gaussians = [multivariate_normal(diff[i, :], std_matrix) for i in range(nims)]

			for cand1 in range(len(x_candidates[ip])):
				pos1 = np.array([x_candidates[ip][cand1], y_candidates[ip][cand1]])
				for cand2 in range(len(x_candidates[ipl])):
					pos2 = np.array([x_candidates[ipl][cand2], y_candidates[ipl][cand2]])
					diff = pos1 - pos2
					rel[cand1, cand2] = np.max([gaussians[i].pdf(diff) for i in range(nims)])
			B_mat[(ip, ipl)] = rel / multivariate_normal([0, 0], std_matrix).pdf([0, 0])

	for (ip, ipl) in B_mat.keys():
		rel = B_mat[(ip, ipl)]
		for i in range(len(x_candidates[ip])):
			rel[i, :] = rel[i, :] / np.sum(rel[i, :])
		B_mat[(ip, ipl)] = rel
	return B_mat


def compute_final_solution_phase_3(xc, yc, probability_map_phase_2, ncandidates, sde, delta, T, edges):
	(height, width, nldms) = probability_map_phase_2.shape
	x_candidates = []  # np.zeros((nldms,ncandidates))
	y_candidates = []  # np.zeros((nldms,ncandidates))

	for i in range(nldms):
		val = -np.sort(-probability_map_phase_2[:, :, i].flatten())[ncandidates]
		if val > 0:
			(y, x) = np.where(probability_map_phase_2[:, :, i] >= val)
		else:
			(y, x) = np.where(probability_map_phase_2[:, :, i] > val)

		if y.size > ncandidates:
			vals = -probability_map_phase_2[y, x, i]
			order = np.argsort(vals)[0:ncandidates]
			y = y[order]
			x = x[order]

		x_candidates.append(x.tolist())
		y_candidates.append(y.tolist())

	b_mat = build_bmat_phase_3(xc, yc, T, x_candidates, y_candidates, edges, sde)

	g = FactorGraph(silent=True)
	nodes = [Variable('x%d' % i, len(x_candidates[i])) for i in range(nldms)]
	for ip in range(nldms):
		for ipl in edges[ip, :].astype(int):
			g.add(Factor('f2_%d_%d' % (ip, ipl), b_mat[(ip, ipl)]))
			g.append('f2_%d_%d' % (ip, ipl), nodes[ip])
			g.append('f2_%d_%d' % (ip, ipl), nodes[ipl])
	for i in range(nldms):
		v = probability_map_phase_2[np.array(y_candidates[i]), np.array(x_candidates[i]), i]
		g.add(Factor('f1_%d' % i, v / np.sum(v)))
		g.append('f1_%d' % i, nodes[i])
	g.compute_marginals()

	x_final = np.zeros(nldms)
	y_final = np.zeros(nldms)

	for i in range(nldms):
		amin = np.argmax(g.nodes['x%d' % i].marginal())
		x_final[i] = x_candidates[i][amin]
		y_final[i] = y_candidates[i][amin]
	return x_final / delta, y_final / delta

def main():
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Landmark Detection', prog='Cytomine Landmark Detection : Landmark Dectector', version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--cytomine_predict_images', type='string', dest='cytomine_predict_images', help='The identifier of the images to predict, separated by a comma (no spaces).')
	p.add_option('--model_load_from', default='/tmp/', type="string", dest="model_load_from", help="The repository where the models are stored")
	p.add_option('--model_name', type="string", dest="cytomine_model_name", help="The name of the model to use for detection")
	p.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	options, arguments = p.parse_args(args=sys.argv)

	parameters = {
		'cytomine_host': options.cytomine_host,
		'cytomine_public_key': options.cytomine_public_key,
		'cytomine_private_key': options.cytomine_private_key,
		'cytomine_id_software': options.cytomine_id_software,
		'cytomine_base_path': options.cytomine_base_path,
		'cytomine_working_path': options.cytomine_working_path.rstrip('/')+'/',
		'cytomine_id_project': options.cytomine_id_project,
		'cytomine_predict_images': options.cytomine_predict_images,
		'model_load_from': options.model_load_from.rstrip('/') + '/',
		'cytomine_model_name': options.cytomine_model_name,
		'image_type': options.image_type,
		'verbose': str2bool(options.verbose)
	}

	parameters['model_load_from'] = parameters['model_load_from'].rstrip('/') + '/'
	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'], parameters['cytomine_public_key'], parameters['cytomine_private_key'], base_path=parameters['cytomine_base_path'], working_path=parameters['cytomine_working_path'], verbose=parameters['verbose'])
	current_user = cytomine_connection.get_current_user()
	repository = parameters['cytomine_working_path'] + str(parameters['cytomine_id_project']) + '/'

	if current_user.algo:
		sys.path.append('/software_router/algo/landmark_model_builder/')
		user_job = current_user
	else:
		sys.path.append('../landmark_model_builder/')
		user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))

	job = cytomine_connection.get_job(user_job.job)
	cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Uploading annotations...")
	clf = joblib.load('%s%s_classifier_phase1.pkl' % (parameters['model_load_from'], parameters['cytomine_model_name']))
	feature_offsets_1 = joblib.load(
		'%s%s_offsets_phase1.pkl' % (parameters['model_load_from'], options.cytomine_model_name))
	par = {}
	F = open('%s%s_dmbl_parameters.conf' % (parameters['model_load_from'], options.cytomine_model_name))
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

	for id_img in pr_im:
		probability_map = probability_map_phase_1(repository, id_img, clf, feature_offsets_1, float(par['model_delta']))
		np.savez_compressed('%spmap_phase1-%s-%d.npy' % (repository, options.cytomine_model_name, id_img), probability_map)

	del clf

	for id_term in id_terms:
		reg = joblib.load('%s%s_dmbl_regressor_phase2_%d.pkl' % (parameters['model_load_from'], options.cytomine_model_name, id_term))
		feature_offsets_2 = joblib.load(
			'%s%s_dmbl_offsets_phase2_%d.pkl' % (parameters['model_load_from'], options.cytomine_model_name, id_term))
		i = 0
		for id_img in pr_im:
			probability_map = np.load('%spmap_phase1-%s-%d.npy.npz' % (repository, options.cytomine_model_name, id_img))['arr_0']
			probability_map_phase_2 = agregation_phase_2(repository, id_img, i, probability_map, reg, float(par['model_delta']), feature_offsets_2, int(par['model_filter_size']), float(par['model_beta']), int(par['model_n_iterations']))
			np.savez_compressed('%spmap_phase2-%s-%d-%d.npy' % (repository, options.cytomine_model_name, id_img, id_term), probability_map_phase_2)
			i += 1

	edges = joblib.load('%s%s_edgematrix_phase3.pkl' % (parameters['model_load_from'], options.cytomine_model_name))
	hash_error = {}
	for id_img in pr_im:
		probability_map_phase_2 = np.load('%spmap_phase2-%s-%d-%d.npy.npz' % (repository, options.cytomine_model_name, id_img, id_terms[0]))['arr_0']
		(h, w) = probability_map_phase_2.shape
		pmap = np.zeros((h, w, len(id_terms)))
		pmap[:, :, 0] = probability_map_phase_2
		i = 1
		for id_term in id_terms[1:]:
			pmap[:, :, i] = np.load('%spmap_phase2-%s-%d-%d.npy.npz' % (repository, options.cytomine_model_name, id_img, id_term))['arr_0']
			i += 1
		x_final, y_final = compute_final_solution_phase_3(xc, yc, pmap, int(par['model_ncandidates']), float(par['model_sde']), float(par['model_delta']), int(par['model_T']), edges)
		i = 0
		for id_term in id_terms:
			circle = Point(x_final[i], h - y_final[i])
			location = circle.wkt
			new_annotation = cytomine_connection.add_annotation(location, id_img)
			cytomine_connection.add_annotation_term(new_annotation.id, term=id_term, expected_term=id_term, rate=1.0, annotation_term_model=models.AlgoAnnotationTerm)
			if id_img in idim_to_i:
				xreal = xc[idim_to_i[id_img], t_to_i[id_term]]
				yreal = yc[idim_to_i[id_img], t_to_i[id_term]]
				er = np.linalg.norm([xreal - x_final[i], yreal - y_final[i]])
				tup = (id_img, id_term)
				if (tup in hash_error and er < hash_error[tup]) or (not (tup in hash_error)):
					hash_error[(id_img, id_term)] = er
			i += 1


if __name__ == "__main__":
	main()
