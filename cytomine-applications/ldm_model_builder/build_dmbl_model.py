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

from SeparateTrees import SeparateTrees
from SeparateTreesRegressor import SeparateTreesRegressor
from ldmtools import *
import numpy as np
from multiprocessing import Pool
import scipy.ndimage as snd
from sklearn.externals import joblib
from download import *
import sys,cytomine
import optparse


def dataset_from_coordinates(img, x, y, feature_offsets):
	(h, w) = img.shape
	original_values = img[y.clip(min=0, max=h - 1), x.clip(min=0, max=w - 1)]
	dataset = np.zeros((x.size, feature_offsets[:, 0].size))
	
	for i in range(feature_offsets[:, 0].size):
		dataset[:, i] = original_values - img[(y + feature_offsets[i, 1]).clip(min=0, max=h - 1), (x + feature_offsets[i, 0]).clip(min=0, max=w - 1)]
	
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

	n_out = int(np.round(P*nroff))
	rep = np.zeros((x.size * nroff) + n_out)
	xs = np.zeros((x.size * nroff) + n_out).astype('int')
	ys = np.zeros((x.size * nroff) + n_out).astype('int')
	for ip in range(x.size):
		xs[ip*nroff:(ip+1)*nroff] = x[ip] + R_offsets[:, 0]
		ys[ip*nroff:(ip+1)*nroff] = y[ip] + R_offsets[:, 1]
		rep[ip*nroff:(ip+1)*nroff] = ip
	mask[ys, xs] = 0
	(ym, xm) = np.where(mask == 1)
	perm = np.random.permutation(ym.size)[0:n_out]
	ym = ym[perm]
	xm = xm[perm]
	xs[x.size * nroff:] = xm
	ys[y.size * nroff:] = ym
	rep[x.size * nroff:] = x.size
	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep


def dataset_mp_helper(jobargs):
	return image_dataset_phase_1(*jobargs)


def get_dataset_phase_1(repository, training_images, image_ids, n_jobs, feature_offsets, R_offsets, delta, P, X, Y):
	p = Pool(n_jobs)
	Xc = np.round(X * delta).astype('int')
	Yc = np.round(Y * delta).astype('int')
	(nims, nldms) = Xc.shape
	jobargs = []
	
	for i in range(nims):
		if image_ids[i] in training_images:
			jobargs.append((repository, image_ids[i], Xc[i, :], Yc[i, :], feature_offsets, R_offsets, delta, P))
	data = p.map(dataset_mp_helper, jobargs)
	p.close()
	p.join()
	(nroff, blc) = R_offsets.shape
	nims = len(training_images)
	
	n_in = nroff*nldms
	n_out = int(np.round(nroff*P))
	n_tot = n_in+n_out
	DATASET = np.zeros((nims * n_tot, feature_offsets[:, 0].size))
	REP = np.zeros(nims * n_tot)
	IMG = np.zeros(nims * n_tot)
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


def build_phase_1_model(repository, tr_image=[], image_ids=[], n_jobs=1, NT=32, F=100, R=2, sigma=10, delta=0.25, P=1, X=None, Y=None):
	std_matrix = np.eye(2) * (sigma ** 2)
	feature_offsets = np.round(np.random.multivariate_normal([0, 0], std_matrix, NT * F)).astype('int')
	R_offsets = []
	
	for x1 in range(-R, R + 1):
		for x2 in range(-R, R + 1):
			if (np.linalg.norm([x1, x2]) <= R):
				R_offsets.append([x1, x2])

	R_offsets = np.array(R_offsets).astype('int')
	(dataset, rep, img) = get_dataset_phase_1(repository, tr_image, image_ids, n_jobs, feature_offsets, R_offsets, delta, P, X, Y)
	return dataset, rep, img, feature_offsets


def probability_map_phase_1(repository, image_number, clf, feature_offsets, delta):
	img = makesize(snd.zoom(readimage(repository, image_number), delta), 1)
	(h, w) = img.shape
	ys = []
	xs = []
	c = np.arange((h - 2) * (w - 2))
	ys = 1 + np.round(c / (w - 2)).astype('int')
	xs = 1 + np.mod(c, (w - 2))
	step = 20000
	b = 0
	probability_map = None
	nldms = -1

	while b < xs.size:

		next_b = min(b + step, xs.size)
		dataset = dataset_from_coordinates(img, xs[b:next_b], ys[b:next_b], feature_offsets)
		probabilities = clf.predict_proba(dataset)

		if (nldms == -1):
			(ns, nldms) = probabilities.shape
			probability_map = np.zeros((h - 2, w - 2, nldms))

		for ip in range(nldms):
			probability_map[ys[b:next_b] - 1, xs[b:next_b] - 1, ip] = probabilities[:, ip]
			
		b = next_b

	return probability_map


def image_dataset_phase_2(repository, image_number, x, y, feature_offsets, R_offsets, delta):
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
	rep = np.zeros((nroff, 2))
	number = image_number
	xs = (x + R_offsets[:, 0]).astype('int')
	ys = (y + R_offsets[:, 1]).astype('int')
	rep[:, 0] = R_offsets[:, 0]
	rep[:, 1] = R_offsets[:, 1]
	dataset = dataset_from_coordinates(img, xs, ys, feature_offsets)
	return dataset, rep, number


def dataset_mp_helper_phase_2(jobargs):
	return image_dataset_phase_2(*jobargs)

def get_dataset_phase_2(repository, tr_images, image_ids, n_jobs, id_term, feature_offsets, R_offsets, delta):
	p = Pool(n_jobs)
	(Xc, Yc, Xp, Yp, ims) = getcoords(repository.rstrip('/') + '/txt/', id_term)
	nims = Xc.size
	jobargs = []
	
	for i in range(nims):
		if image_ids[i] in tr_images:
			jobargs.append((repository, image_ids[i], Xc[i], Yc[i], feature_offsets, R_offsets, delta))
	
	data = p.map(dataset_mp_helper_phase_2, jobargs)
	p.close()
	p.join()
	(nroff, blc) = R_offsets.shape
	nims = len(tr_images)
	DATASET = np.zeros((nims * nroff, feature_offsets[:, 0].size))
	REP = np.zeros((nims * nroff, 2))
	NUMBER = np.zeros(nims * nroff)
	b = 0
	
	for (d, r, n) in data:
		(nd, nw) = d.shape
		DATASET[b:b + nd, :] = d
		REP[b:b + nd, :] = r
		NUMBER[b:b + nd] = n
		b = b + nd

	DATASET = DATASET[0:b, :]
	REP = REP[0:b]
	NUMBER = NUMBER[0:b]
	return DATASET, REP, NUMBER

def build_phase_2_model(repository, tr_image=None, image_ids=None, n_jobs=1, IP=0, NT=32, F=100, R=3, N=500, sigma=10, delta=0.25):
	std_matrix = np.eye(2) * (sigma ** 2)
	feature_offsets = np.round(np.random.multivariate_normal([0, 0], std_matrix, NT * F)).astype('int')
	R_offsets = np.zeros((N, 2))
	dis = np.random.ranf(N) * R
	ang = np.random.ranf(N) * 2 * np.pi
	R_offsets[:, 0] = np.round((dis * np.cos(ang))).astype('int')
	R_offsets[:, 1] = np.round((dis * np.sin(ang))).astype('int')
	(dataset, rep, number) = get_dataset_phase_2(repository, tr_image, image_ids, n_jobs, IP, feature_offsets, R_offsets, delta)
	return dataset, rep, number, feature_offsets

def build_edgematrix_phase_3(Xc, Yc, sde, delta, T):
	
	Xc = Xc * delta
	Yc = Yc * delta
	(nims, nldms) = Xc.shape
	differential_entropy = np.eye(nldms) + np.inf
	c1 = np.zeros((nims, 2))
	c2 = np.zeros((nims, 2))

	for ldm1 in range(nldms):
		c1[:, 0] = Xc[:, ldm1]
		c1[:, 1] = Yc[:, ldm1]
		for ldm2 in range(ldm1 + 1, nldms):
			c2[:, 0] = Xc[:, ldm2]
			c2[:, 1] = Yc[:, ldm2]
			diff = c1 - c2
			d = diff - np.mean(diff, axis=0)
			d = np.mean(np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2)))
			differential_entropy[ldm1, ldm2] = d
			differential_entropy[ldm2, ldm1] = d

	edges = np.zeros((nldms, T))

	for ldm in range(nldms):
		edges[ldm, :] = np.argsort(differential_entropy[ldm, :])[0:T]

	return edges.astype(int)


def main():
	p = optparse.OptionParser(description='Cytomine Landmark Detection : Model building', prog='Cytomine Landmark Detector : Model builder', version='0.1')
	p.add_option('--cytomine_host', type="string", default='beta.cytomine.be', dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
	p.add_option('--cytomine_public_key', type="string", default='XXX', dest="cytomine_public_key", help="Cytomine public key")
	p.add_option('--cytomine_private_key', type="string", default='YYY', dest="cytomine_private_key", help="Cytomine private key")
	p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software", help="The Cytomine software identifier")
	p.add_option('--cytomine_base_path', type="string", default='/api/', dest="cytomine_base_path", help="Cytomine base path")
	p.add_option('--cytomine_working_path', default="/tmp/", type="string", dest="cytomine_working_path", help="The working directory (eg: /tmp)")
	p.add_option('--cytomine_training_images', default="all", type="string", dest="cytomine_training_images", help="identifiers of the images used to create the models. ids must be separated by commas (no spaces). If 'all' is mentioned instead, every image with manual annotation will be used.")
	p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
	p.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
	p.add_option('--model_njobs', type='int', default=1, dest='model_njobs', help="The number of processors used for model building")
	p.add_option('--cytomine_id_terms', type='string', default=1, dest='cytomine_id_terms', help="The identifiers of the terms to create detection models for. Terms must be separated by commas (no spaces). If 'all' is mentioned instead, every terms will be detected.")
	p.add_option('--model_NT_P1', type='int', default=6, dest='model_NT_P1', help="Number of trees for phase 1.")
	p.add_option('--model_F_P1', type='int', default=200, dest='model_F_P1', help="Number of features for phase 1.")
	p.add_option('--model_R_P1', type='int', default=3, dest='model_R_P1', help="Radius in which phase 1 samples are extracted.")
	p.add_option('--model_sigma', type='int', default=20, dest='model_sigma', help="Standard deviation for the gaussian.")
	p.add_option('--model_delta', type='float', default=3, dest='model_delta', help="Resizing factor.")
	p.add_option('--model_P', type='float', default=3, dest='model_P', help="Proportion of non-landmarks.")
	p.add_option('--model_R_P2', type='int', default=3, dest='model_R_P2', help="Radius in which phase 2 samples are extracted.")
	p.add_option('--model_ns_P2', type='int', default=3, dest='model_ns_P2', help="Number of samples for phase 2.")
	p.add_option('--model_NT_P2', type='int', default=3, dest='model_NT_P2', help="Number of trees for phase 2.")
	p.add_option('--model_F_P2', type='int', default=3, dest='model_F_P2', help="Number of features for phase 2.")
	p.add_option('--model_filter_size', type='int', default=3, dest='model_filter_size', help="Size of the filter for phase 2.")
	p.add_option('--model_beta', type='float', default=3, dest='model_beta', help="Beta for phase 2.")
	p.add_option('--model_n_iterations', type='int', default=3, dest='model_n_iterations', help="Number of iterations for phase 2.")
	p.add_option('--model_ncandidates', type='int', default=3, dest='model_ncandidates', help="Number of candidates for phase 3.")
	p.add_option('--model_sde', type='float', default=10., dest='model_sde', help="Standard deviation for gaussian phase 3.")
	p.add_option('--model_T', type='int', default=3, dest='model_T', help="Number of edges for phase 3.")
	p.add_option('--model_save_to', type='string', default='/tmp/', dest='model_save_to', help="Destination for model storage")
	p.add_option('--model_name', type='string', dest='model_name', help="Name of the model (used for saving)")
	p.add_option('--verbose', type="string", default="0", dest="verbose", help="Turn on (1) or off (0) verbose mode")

	opt_parser, arguments = p.parse_args(args=sys.argv)
	opt_parser.cytomine_working_path = opt_parser.cytomine_working_path.rstrip('/') + '/'
	cytomine_connection = cytomine.Cytomine(opt_parser.cytomine_host, opt_parser.cytomine_public_key, opt_parser.cytomine_private_key, base_path=opt_parser.cytomine_base_path, working_path=opt_parser.cytomine_working_path, verbose=str2bool(opt_parser.verbose))
	current_user = cytomine_connection.get_current_user()

	if not current_user.algo:
		user_job = cytomine_connection.add_user_job(opt_parser.cytomine_id_software, opt_parser.cytomine_id_project)
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))
	else:
		user_job = current_user

	run_by_user_job = True
	job = cytomine_connection.get_job(user_job.job)
	cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Bulding model...")
	job_parameters = {}
	job_parameters['cytomine_id_terms'] = opt_parser.cytomine_id_terms
	job_parameters['model_njobs'] = opt_parser.model_njobs
	job_parameters['model_NT_P1'] = opt_parser.model_NT_P1
	job_parameters['model_F_P1'] = opt_parser.model_F_P1
	job_parameters['model_R_P1'] = opt_parser.model_R_P1
	job_parameters['model_sigma'] = opt_parser.model_sigma
	job_parameters['model_delta'] = opt_parser.model_delta
	job_parameters['model_P'] = opt_parser.model_P
	job_parameters['model_R_P2'] = opt_parser.model_R_P2
	job_parameters['model_ns_P2'] = opt_parser.model_ns_P2
	job_parameters['model_NT_P2'] = opt_parser.model_NT_P2
	job_parameters['model_F_P2'] = opt_parser.model_F_P2
	job_parameters['model_filter_size'] = opt_parser.model_filter_size
	job_parameters['model_beta'] = opt_parser.model_beta
	job_parameters['model_n_iterations'] = opt_parser.model_n_iterations
	job_parameters['model_ncandidates'] = opt_parser.model_ncandidates
	job_parameters['model_sde'] = opt_parser.model_sde
	job_parameters['model_T'] = opt_parser.model_T
	model_repo = opt_parser.model_save_to

	if not os.path.isdir(model_repo):
		os.mkdir(model_repo)

	if not run_by_user_job:
		cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(opt_parser.cytomine_id_software), job_parameters)

	download_images(cytomine_connection, opt_parser.cytomine_id_project)
	download_annotations(cytomine_connection, opt_parser.cytomine_id_project, opt_parser.cytomine_working_path)
	repository = opt_parser.cytomine_working_path + str(opt_parser.cytomine_id_project) + '/'
	(xc, yc, xr, yr, ims, term_to_i, i_to_term) = getallcoords(repository.rstrip('/') + '/txt/')
	(nims, nldms) = xc.shape

	if opt_parser.cytomine_id_terms != 'all':
		term_list = np.sort([int(term) for term in opt_parser.cytomine_id_terms.split(',')])
		Xc = np.zeros((nims, len(term_list)))
		Yc = np.zeros(Xc.shape)
		i = 0
		for id_term in term_list:
			Xc[:, i] = xc[:, term_to_i[id_term]]
			Yc[:, i] = yc[:, term_to_i[id_term]]
			i += 1
	else:
		term_list = np.sort(term_to_i.keys())
		Xc = xc
		Yc = yc

	if opt_parser.cytomine_training_images == 'all':
		tr_im = ims
	else:
		tr_im = [int(p) for p in opt_parser.cytomine_training_images.split(',')]

	(dataset, rep, img, feature_offsets_1) = build_phase_1_model(repository, tr_image=tr_im, image_ids=ims, n_jobs=opt_parser.model_njobs, NT=opt_parser.model_NT_P1, F=opt_parser.model_F_P1, R=opt_parser.model_R_P1, sigma=opt_parser.model_sigma, delta=opt_parser.model_delta, P=opt_parser.model_P, X=Xc, Y=Yc)
	clf = SeparateTrees(n_estimators=opt_parser.model_NT_P1, n_jobs=opt_parser.model_njobs)
	clf = clf.fit(dataset, rep)
	joblib.dump(clf, '%s%s_classifier_phase1.pkl' % (model_repo, opt_parser.model_name))
	joblib.dump(feature_offsets_1, '%s%s_offsets_phase1.pkl' % (model_repo, opt_parser.model_name))

	for id_term in term_list:
		(dataset, rep, number, feature_offsets_2) = build_phase_2_model(repository, tr_image=tr_im, image_ids=ims, n_jobs=opt_parser.model_njobs, IP=id_term, NT=opt_parser.model_NT_P2, F=opt_parser.model_F_P2, R=opt_parser.model_R_P2, N=opt_parser.model_ns_P2, sigma=opt_parser.model_sigma, delta=opt_parser.model_delta)
		reg = SeparateTreesRegressor(n_estimators=opt_parser.model_NT_P2, n_jobs=opt_parser.model_njobs)
		reg.fit(dataset, rep)
		joblib.dump(reg, '%s%s_dmbl_regressor_phase2_%d.pkl' % (model_repo, opt_parser.model_name, id_term))
		joblib.dump(feature_offsets_2, '%s%s_dmbl_offsets_phase2_%d.pkl' % (model_repo, opt_parser.model_name, id_term))

	(nims, nldms) = xc.shape
	X = np.zeros((len(tr_im), nldms))
	Y = np.zeros(X.shape)
	j = 0
	
	for i in range(nims):
		if ims[i] in tr_im:
			X[j, :] = xc[i, :]
			Y[j, :] = yc[i, :]
			j += 1

	edges = build_edgematrix_phase_3(X, Y, opt_parser.model_sde, opt_parser.model_delta, opt_parser.model_T)
	joblib.dump(edges, '%s%s_edgematrix_phase3.pkl' % (opt_parser.model_save_to, opt_parser.model_name))
	F = open('%s%s_dmbl_parameters.conf' % (opt_parser.model_save_to, opt_parser.model_name), 'wb')
	F.write('cytomine_id_terms %s\n' % opt_parser.cytomine_id_terms)
	F.write('model_njobs %d\n' % opt_parser.model_njobs)
	F.write('model_NT_P1 %d\n' % opt_parser.model_NT_P1)
	F.write('model_F_P1 %d\n' % opt_parser.model_F_P1)
	F.write('model_R_P1 %d\n' % opt_parser.model_R_P1)
	F.write('model_sigma %f\n' % opt_parser.model_sigma)
	F.write('model_delta %f\n' % opt_parser.model_delta)
	F.write('model_P %f\n' % opt_parser.model_P)
	F.write('model_R_P2 %d\n' % opt_parser.model_R_P2)
	F.write('model_ns_P2 %d\n' % opt_parser.model_ns_P2)
	F.write('model_NT_P2 %d\n' % opt_parser.model_NT_P2)
	F.write('model_F_P2 %d\n' % opt_parser.model_F_P2)
	F.write('model_filter_size %d\n' % opt_parser.model_filter_size)
	F.write('model_beta %f\n' % opt_parser.model_beta)
	F.write('model_n_iterations %d\n' % opt_parser.model_n_iterations)
	F.write("model_ncandidates %d\n" % opt_parser.model_ncandidates)
	F.write('model_sde %f\n' % opt_parser.model_sde)
	F.write('model_T %d' % opt_parser.model_T)
	F.close()

if __name__ == "__main__":
	main()
