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

import sys
from sklearn.externals import joblib
from shapely.geometry import Point
import optparse
from cytomine import cytomine, models
from download import *
from ldmtools import *
from build_generic_model import build_dataset_image


"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""


def searchpoint_cytomine(repository, current, clf, mx, my, cm, depths, window_size, image_type, npred, feature_type, feature_parameters):
	simage = readimage(repository, current, image_type)
	(height, width) = simage.shape

	P = np.random.multivariate_normal([mx, my], cm, npred)
	x_v = np.round(P[:, 0] * width)
	y_v = np.round(P[:, 1] * height)

	height -= 1
	width -= 1

	n = len(x_v)
	pos = 0

	maxprob = -1
	maxx = []
	maxy = []

	# maximum number of points considered at once in order to not overload the
	# memory.
	step = 100000

	for index in range(len(x_v)):
		xv = x_v[index]
		yv = y_v[index]
		if xv < 0:
			x_v[index] = 0
		if yv < 0:
			y_v[index] = 0
		if xv > width:
			x_v[index] = width
		if yv > height:
			y_v[index] = height

	while pos < n:
		xp = np.array(x_v[pos:min(n, pos + step)])
		yp = np.array(y_v[pos:min(n, pos + step)])
		DATASET = build_dataset_image(simage, window_size, xp, yp, feature_type, feature_parameters, depths)
		pred = clf.predict_proba(DATASET)
		pred = pred[:, 1]
		maxpred = np.max(pred)
		if maxpred >= maxprob:
			positions = np.where(pred == maxpred)
			positions = positions[0]
			xsup = xp[positions]
			ysup = yp[positions]
			if maxpred > maxprob:
				maxprob = maxpred
				maxx = xsup
				maxy = ysup
			else:
				maxx = np.concatenate((maxx, xsup))
				maxy = np.concatenate((maxy, ysup))
		pos += step

	return np.median(maxx), (height + 1) - np.median(maxy), np.median(maxy)


if __name__ == "__main__":

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
	p.add_option('--model_names', type="string", dest="cytomine_model_names", help="The names of the models to use for detection (separated by commas, no spaces)")
	p.add_option('--image_type', type='string', default='jpg', dest='image_type', help="The type of the images that will be used (jpg, bmp, png,...)")
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
		'model_load_from': options.model_load_from,
		'cytomine_model_names': options.cytomine_model_names,
		'image_type': options.image_type,
		'verbose': str2bool(options.verbose)
	}
	parameters['cytomine_working_path'] = parameters['cytomine_working_path'].rstrip('/')+'/'
	parameters['model_load_from'] = parameters['model_load_from'].rstrip('/')+'/'

	cytomine_connection = cytomine.Cytomine(parameters['cytomine_host'], parameters['cytomine_public_key'], parameters['cytomine_private_key'], base_path=parameters['cytomine_base_path'], working_path=parameters['cytomine_working_path'], verbose=parameters['verbose'])

	current_user = cytomine_connection.get_current_user()
	run_by_user_job = False
	if current_user.algo:
		sys.path.append('/software_router/algo/landmark_model_builder/')
		user_job = current_user
		run_by_user_job = True
	else:
		sys.path.append('../landmark_model_builder/')
		user_job = cytomine_connection.add_user_job(parameters['cytomine_id_software'], parameters['cytomine_id_project'])
		cytomine_connection.set_credentials(str(user_job.publicKey), str(user_job.privateKey))

	download_images(cytomine_connection, int(parameters['cytomine_id_project']))

	repository = parameters['cytomine_working_path'] + str(parameters['cytomine_id_project']) + '/'

	model_repo = parameters['model_load_from']
	model_names = parameters['cytomine_model_names'].split(',')
	nmodels = len(model_names)
	image_type = parameters['image_type']
	id_software = parameters['cytomine_id_software']
	coords = {}
	ips = []

	feature_types = []

	job = cytomine_connection.get_job(user_job.job)
	job = cytomine_connection.update_job_status(job, status=job.RUNNING, progress=0, status_comment="Uploading annotations...")

	progress = 0
	delta = 90 / len(model_names)

	repository = options.cytomine_working_path + str(options.cytomine_id_project) + '/'
	txt_repository = options.cytomine_working_path + '%d/txt/' % options.cytomine_id_project
	(xc, yc, xr, yr, ims, t_to_i, i_to_t) = getallcoords(txt_repository)
	pr_im = None

	if options.cytomine_predict_images == 'all':
		pr_im = ims
	else:
		pr_im = [int(id_im) for id_im in options.cytomine_predict_images.split(',')]

	idim_to_i = {}
	for i in range(len(ims)):
		idim_to_i[ims[i]] = i

	hash_term = {}
	hash_error = {}
	for model in model_names:
		F = open('%s%s.conf' % (model_repo, model))
		par = {}
		for l in F.readlines():
			line = l.rstrip('\n')
			tab = line.split(' ')
			par[tab[0]] = tab[1]

		if par['cytomine_id_terms'] == 'all':
			term_list = t_to_i.keys()
		else:
			term_list = [int(term) for term in par['cytomine_id_terms'].split(',')]

		ips.append(par['cytomine_id_terms'])
		feature_types.append(par['feature_type'])

		for id_term in term_list:
			if id_term in hash_term:
				print "Term %d was already predicted by another model. Only best prediction is kept for result matrix."
			else:
				hash_term[id_term] = 1

			mx, my, cm = joblib.load('%s%s_%d_cov.pkl' % (model_repo, model, id_term))
			clf = joblib.load('%s%s_%d.pkl' % (model_repo, model, id_term))
			if par['feature_type'] == 'haar':
				fparams = joblib.load('%s%s_%d_fparameters.pkl' % (model_repo, model, id_term))
			elif par['feature_type'] == 'gaussian':
				fparams = joblib.load('%s%s_%d_fparameters.pkl' % (model_repo, model, id_term))
			else:
				fparams = None

			progress += delta

			for j in pr_im:
				(x, y, y_o) = searchpoint_cytomine(repository, j, clf, mx, my, cm, 1. / (2. ** np.arange(int(par['model_depth']))), int(par['window_size']), image_type, int(par['model_npred']), par['feature_type'], fparams)
				circle = Point(x, y)
				location = circle.wkt
				new_annotation = cytomine_connection.add_annotation(location, j)
				cytomine_connection.add_annotation_term(new_annotation.id, term=id_term, expected_term=id_term, rate=1.0, annotation_term_model=models.AlgoAnnotationTerm)
				if j in idim_to_i:
					xreal = xc[idim_to_i[j], t_to_i[id_term]]
					yreal = yc[idim_to_i[j], t_to_i[id_term]]
					er = np.linalg.norm([xreal-x, yreal-y_o])
					tup = (j, id_term)
					if (tup in hash_error and er < hash_error[tup]) or (not (tup in hash_error)):
						hash_error[(j, id_term)] = er
		F.close()

	job = cytomine_connection.update_job_status(job, status=job.TERMINATED, progress=100, status_comment="Annotations uploaded!")

	csv_line = ";"
	id_terms = hash_term.keys()
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

	job_parameters = {'cytomine_models': parameters['cytomine_model_names'], 'cytomine_predict_images': parameters['cytomine_predict_images'], 'prediction_error': csv_line}
	job_parameters_values = cytomine_connection.add_job_parameters(user_job.job, cytomine_connection.get_software(id_software), job_parameters)
