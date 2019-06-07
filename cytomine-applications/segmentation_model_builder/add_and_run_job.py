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

__author__ = "Marée Raphael <raphael.maree@ulg.ac.be>"
__contributors__ = ["Stévens Benjamin <b.stevens@ulg.ac.be>", "Elodie Burtin <elodie.burtin@cytomine.coop>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"
__version__ = "2.0.0"

import os
import shutil
import sys
import logging
import time

import cPickle as pickle
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

import cytomine
from cytomine.models import *
from pyxit.estimator import PyxitClassifier, _get_output_from_mask

# Usage:
# This file download (dump) existing annotations from the server at specified dump_zoom
# It builds a segmentation model (using randomly extracted subwindows and extra-trees with multiple outputs) that tries
# to discriminate between the predicted_terms (regrouped into one class),and all other terms (regrouped in a second
# class), but without using terms specified in excluded_terms.
# You need to specify the Cytomine id_project, and the software id (as produced by the add_software.py script)
# See test-train.sh


def run(cyto_job, parameters):
    logging.info("----- segmentation_model_builder v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job

    projects = map(int, parameters.cytomine_annotation_projects.split(','))
    predict_terms = map(int, parameters.cytomine_predict_terms.split(','))
    excluded_terms = parameters.cytomine_excluded_terms
    excluded_terms = map(int, excluded_terms.split(',')) if excluded_terms and excluded_terms != "null" else []

    working_path = os.path.join("tmp", str(job.id))
    if not os.path.exists(working_path):
        logging.info("Creating annotation directory: %s", working_path)
        os.makedirs(working_path)

    try:
        # Get annotation data
        job.update(statusComment="Fetching data")

        # Retrieve annotations from each annotation project
        annotations = []
        for prj in projects:
            logging.info("Retrieving annotations in project %d", prj)
            annotations_prj = AnnotationCollection(project=prj, showTerm=True, reviewed=parameters.cytomine_reviewed).fetch()
            logging.info("# annotations in project %d: %d", prj, len(annotations_prj))

            annotations += annotations_prj

        nb_annots = len(annotations)
        logging.info("# total annotations: %d", nb_annots)

        terms = TermCollection().fetch_with_filter("project", cyto_job.project.id)
        # Change initial problem into binary problem : "predict_terms" vs others
        map_classes = {term.id: int(term.id in predict_terms) for term in terms}
        for term in excluded_terms:
            map_classes[term] = -1  # excluded class

        classes = [0, 1]
        dest_patterns = {cls: os.path.join(working_path, str(cls), "{image}_{id}.png") for cls in classes}

        x = []
        y = []
        for (i, annot) in enumerate(annotations):
            job.update(progress=int(40*i / nb_annots), statusComment="Treating annotation {}/{}".format(i, nb_annots))

            class_annot = 0
            terms = annot.term if annot.term is not None else []
            for term in terms:
                class_annot = map_classes[term]
                if class_annot != 0:
                    break

            if class_annot == -1:  # excluded => do not dump and do not add to dataset
                continue

            annot.dump(dest_patterns[class_annot], mask=True, alpha=True, zoom=parameters.cytomine_zoom_level)
            x.append(annot.filename)
            y.append(class_annot)

        x = np.array(x)
        y = np.array(y)
        logging.debug("X length: %d", len(x))
        logging.debug("y length: %d", len(y))

        # Instantiate classifiers
        job.update(statusComment="[pyxit.main] Initializing PyxitClassifier...", progress=40)
        forest = ExtraTreesClassifier(n_estimators=parameters.forest_n_estimators,
                                      max_features=parameters.forest_max_features,
                                      min_samples_split=parameters.forest_min_samples_split,
                                      n_jobs=parameters.pyxit_n_jobs,
                                      verbose=True)

        pyxit = PyxitClassifier(base_estimator=forest,
                                n_subwindows=parameters.pyxit_n_subwindows,
                                min_size=0.0,  # segmentation use fixed-size subwindows
                                max_size=1.0,  # segmentation use fixed-size subwindows
                                target_width=parameters.pyxit_target_width,
                                target_height=parameters.pyxit_target_height,
                                interpolation=parameters.pyxit_interpolation,
                                transpose=parameters.pyxit_transpose,
                                colorspace=parameters.pyxit_colorspace,
                                fixed_size=parameters.pyxit_fixed_size,
                                n_jobs=parameters.pyxit_n_jobs,
                                verbose=True,
                                get_output=_get_output_from_mask)

        if parameters.pyxit_save_to:
            d = os.path.dirname(parameters.pyxit_save_to)
            if not os.path.exists(d):
                os.makedirs(d)
            fd = open(parameters.pyxit_save_to, "wb")
            pickle.dump(classes, fd, protocol=pickle.HIGHEST_PROTOCOL)

        job.update(statusComment="[pyxit.main] Extracting {} subwindows from each image in {}".format(
            parameters.pyxit_n_subwindows, working_path), progress=50)
        # Extract random subwindows in dumped annotations
        _X, _y = pyxit.extract_subwindows(x, y)

        # Build pixel classifier
        job.update(statusComment="[pyxit.main] Fitting Pyxit Segmentation Model", progress=75)
        logging.info("Start fitting Pyxit segmentation model")
        start = time.time()
        pyxit.fit(x, y, _X=_X, _y=_y)
        end = time.time()
        logging.debug("Elapsed time FIT: %d s", end-start)

        logging.debug("pyxit.base_estimator.n_classes_: %s", pyxit.base_estimator.n_classes_)
        logging.debug("pyxit.base_estimator.classes_: %s", pyxit.base_estimator.classes_)

        if parameters.pyxit_save_to:
            logging.debug("----------------------------------------------------------------")
            logging.debug("[pyxit.main] Saving Pyxit Segmentation Model locally into %s", parameters.pyxit_save_to)
            logging.debug("----------------------------------------------------------------")

            pickle.dump(pyxit, fd, protocol=pickle.HIGHEST_PROTOCOL)
            fd.close()

            # job_data = JobData(job.id, "Model", "model.pkl").save()
            # job_data.upload(parameters.pyxit_save_to)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)

        logging.debug("Leaving run()")


if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)
