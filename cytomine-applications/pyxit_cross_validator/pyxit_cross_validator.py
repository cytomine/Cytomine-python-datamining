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

__author__          = "Mormont Romain <r.mormont@ulg.ac.be>"
__contributors__    = []
__copyright__       = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

import optparse
import os
import pickle

import numpy as np
import sys

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import LeavePGroupsOut, cross_val_score, GridSearchCV
from sklearn.utils import check_random_state
from sklearn.ensemble import ExtraTreesClassifier

from util import str2bool, mk_window_size_tuples, accuracy_scoring, print_cm, recall_scoring, str2list, Logger, default
from mapper import BinaryMapper, TernaryMapper
from adapters import PyxitClassifierAdapter, SVMPyxitClassifierAdapter
from cytomine import Cytomine
from cytomine.models import Annotation


__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def pickle_pyxit_adapter(pyxit, file_):
    """Pickle the given pyxit model in the given file

    Parameters
    ----------
    pyxit: PyxitClassifierAdapter
        The classifier to pickle
    file_: file
        A file handle
    """
    svm = isinstance(pyxit, SVMPyxitClassifierAdapter)
    pickle.dump("svm" if svm else "normal", file_)
    pickle.dump(pyxit.classes_, file_)
    pickle.dump(pyxit.equivalent_pyxit(), file_)
    if svm:
        pickle.dump(pyxit.svm, file_)


def mk_pyxit(params, random_state=0):
    """Build a scikit-learn compatible model according to the script parameters

    Parameters:
    -----------
    params:
        A command-line parameter
    random_state:
        The random state

    Return
    ------
    pyxit: PyxitClassifierAdapter
        The pyxit classifier
    """
    random_state = check_random_state(random_state)
    forest = ExtraTreesClassifier(
        n_estimators=params.forest_n_estimators,
        max_features=params.forest_max_features[0],
        min_samples_split=params.forest_min_samples_split[0],
        n_jobs=params.n_jobs,
        verbose=False,
        random_state=random_state
    )

    if params.svm:
        return SVMPyxitClassifierAdapter(
            base_estimator=forest,
            C=params.svm_c[0],
            n_subwindows=params.pyxit_n_subwindows,
            min_size=params.pyxit_min_size[0],
            max_size=params.pyxit_max_size[0],
            target_width=params.pyxit_target_width,
            target_height=params.pyxit_target_height,
            interpolation=params.pyxit_interpolation,
            transpose=params.pyxit_transpose,
            colorspace=params.pyxit_colorspace[0],
            fixed_size=params.pyxit_fixed_size,
            n_jobs=params.n_jobs,
            verbose=params.verbose,
            random_state=random_state
        )
    else:
        return PyxitClassifierAdapter(
            base_estimator=forest,
            n_subwindows=params.pyxit_n_subwindows,
            min_size=params.pyxit_min_size[0],
            max_size=params.pyxit_max_size[0],
            target_width=params.pyxit_target_width,
            target_height=params.pyxit_target_height,
            interpolation=params.pyxit_interpolation,
            transpose=params.pyxit_transpose,
            colorspace=params.pyxit_colorspace[0],
            fixed_size=params.pyxit_fixed_size,
            n_jobs=params.n_jobs,
            verbose=params.verbose,
            random_state=random_state
        )


def mk_dataset(params, logger):
    """Fetch the image

    Parameters
    ----------
    params:
        Command line parameters
    logger: Logger
        A logger for print messages

    Returns
    -------
    paths: list (size: N)
        A list containing the path of the dumped crops
    terms: list (size: N)
        The terms associated with the dumped crops
    images: list (size: N)
        The slides ids of each crop
    """
    cytomine = Cytomine(
        params.cytomine_host,
        params.cytomine_public_key,
        params.cytomine_private_key,
        base_path=params.cytomine_base_path,
        working_path=params.cytomine_working_path,
        verbose=False
    )

    # fetch annotation and filter them
    annotations = cytomine.get_annotations(
        id_project=params.cytomine_id_project,
        showMeta=True,
        id_user=params.cytomine_selected_users
    )

    # add reviewed if requested
    if params.cytomine_include_reviewed:
        if len(params.cytomine_reviewed_images) > 0:
            annotations += cytomine.get_annotations(
                id_project=params.cytomine_id_project,
                id_user=params.cytomine_reviewed_users,
                id_image=params.cytomine_reviewed_images,
                showMeta=True,
                reviewed_only=True
            )
        else:
            annotations += cytomine.get_annotations(
                id_project=params.cytomine_id_project,
                id_user=params.cytomine_reviewed_users,
                showMeta=True,
                reviewed_only=True
            )

    logger.log("Number of fetched annotations (not filtered): {}...".format(len(annotations)))

    # Filter annotations frm user criterion
    excluded_set = set(params.cytomine_excluded_annotations)
    excluded_terms = set(params.cytomine_excluded_terms)
    excluded_images = set(params.cytomine_excluded_images)
    annotations.data()[:] = [a for a in annotations
                             if len(a.term) > 0
                                and a.id not in excluded_set
                                and set(a.term).isdisjoint(excluded_terms)
                                and a.image not in excluded_images]

    logger.log("Number of filtered annotations: {}...".format(len(annotations)))

    # dump annotations
    filtered = cytomine.dump_annotations(
        annotations=annotations,
        dest_path=params.pyxit_dir_ls,
        get_image_url_func=Annotation.get_annotation_alpha_crop_url,
        desired_zoom=params.cytomine_zoom_level
    )

    logger.log("Number of dumped annotations: {}...".format(len(filtered)))

    # make file names
    for annotation in filtered:
        if not hasattr(annotation, "filename"):
            annotation.filename = os.path.join(
                params.pyxit_dir_ls,
                annotation.term[0],
                "{}_{}.png".format(annotation.image, annotation.id)
            )

    return zip(*[(annotation.filename, annotation.term[0], annotation.image) for annotation in filtered])


def train_test_split(X, y, labels, test_set_labels):
    """Split the given dataset based on instances' labels
    Parameters
    ----------
    X: iterable
        Training data
    y: iterable
        Predictions
    labels: iterable
        Labels associated with the instances
    test_set_labels: iterable
        A list of labels

    Returns
    -------
    X_train:
    y_train:
    labels_train:
    X_test:
    y_test:
    labels_test:
    """
    np_x, np_y, np_labels = np.array(X), np.array(y), np.array(labels)
    ts = np.in1d(labels, test_set_labels)
    ls = np.logical_not(ts)
    return np_x[ls], np_y[ls], np_labels[ls], np_x[ts], np_y[ts], np_labels[ts]


def score(pyxit, X, y, labels, p, scoring, n_jobs=1, verbose=True):
    """Compute the score by cross-validation of the given model on the passed data
    Parameters
    ----------
    pyxit: PyxitClassifierAdapter
    X: iterable
    y: iterable
    labels: iterable
    p: int
        Number of labels to leave out for each cross-validation iteration
    scoring: callable
        A scoring function with the given interface: scoring(model, X, y)
    n_jobs: int
        Number of jobs available for the
    verbose: bool
    """
    X = np.array(X)
    y = np.array(y)
    verbose = 10 if verbose else 0
    return cross_val_score(
        pyxit, X, y,
        groups=labels,
        scoring=scoring,
        cv=LeavePGroupsOut(p),
        n_jobs=n_jobs,
        verbose=verbose
    )


def where_in(needles, haystack):
    """Identifies which items in haystack are in needles
    Parameters
    ----------
    needles: ndarray
        The numpy array containing the items to look for
    haystack: ndarray
        The numpy array containing the elements to be looked for

    Returns
    -------
    in: ndarray
        The indexes of the items of haystack that are in needles
    out: ndarray
        The indexes of the items that are not
    """
    boolean_mask = np.in1d(haystack, needles)
    return np.where(boolean_mask)[0].astype("int"), np.where(np.logical_not(boolean_mask))[0].astype("int")


def main(argv):
    p = optparse.OptionParser(description="Pyxit/Cytomine Classification Cross Validator", prog="PyXit Classification Cross Validator (PYthon piXiT CV)")

    # Generic cytomine parameters
    p.add_option("--cytomine_host", type="string", default="", dest="cytomine_host", help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option("--cytomine_public_key", type="string", default="", dest="cytomine_public_key", help="Cytomine public key")
    p.add_option("--cytomine_private_key", type="string", default="", dest="cytomine_private_key", help="Cytomine private key")
    p.add_option("--cytomine_base_path", type="string", default="/api/", dest="cytomine_base_path", help="Cytomine base path")
    p.add_option("--cytomine_working_path", type="string", default="/tmp/", dest="cytomine_working_path", help="The working directory where temporary files can be stored (eg: /tmp)")
    p.add_option("--cytomine_zoom_level", "-z", type="int", default=0, dest="cytomine_zoom_level", help="Zoom level for image extraction (0 for no zoom)")
    p.add_option("--cytomine_id_project", type="int", dest="cytomine_id_project", help="The Cytomine project identifier")
    p.add_option("--cytomine_id_software", type="int", dest="cytomine_id_software", help="The Cytomine software identifier")

    # Filtering images, annotations and terms?
    p.add_option("--cytomine_users", type="string", default="", dest="cytomine_selected_users", help="Users of which the annotations should be extracted.")
    p.add_option("--cytomine_excluded_terms", type="string", default="", dest="cytomine_excluded_terms", help="Some terms to exclude. Those terms shouldn't be used for binary of ternary classification groups.")
    p.add_option("--cytomine_excluded_annotations", type="string", default="", dest="cytomine_excluded_annotations", help="Some annotations to exclude.")
    p.add_option("--cytomine_excluded_images", type="string", default="", dest="cytomine_excluded_images", help="Some images to exclude.")

    # Include reviewed ?
    p.add_option("--cytomine_include_reviewed", type="string", default="False", dest="cytomine_include_reviewed", help="True for including reviewed annotations.")
    p.add_option("--cytomine_reviewed_users", type="string", default="", dest="cytomine_reviewed_users", help="For getting only annotations produced by some specific reviewers. No value indicates no filtering.")
    p.add_option("--cytomine_reviewed_images", type="string", default="", dest="cytomine_reviewed_images", help="For getting only annotations produced on some specific images. No value indicate no filtering.")

    # Binary mapping
    p.add_option("--cytomine_binary", type="string", default="False", dest="cytomine_binary", help="Enable binary mapping.")
    p.add_option("--cytomine_positive_terms", type="string", dest="cytomine_positive_terms", help="The terms to map in the positive class (ignored if cytomine_binary is false)")
    p.add_option("--cytomine_negative_terms", type="string", dest="cytomine_negative_terms", help="The terms to map in the negative class (ignored if cytomine_binary is false).")

    # Ternary mapping
    p.add_option("--cytomine_ternary", type="string", default="False", dest="cytomine_ternary", help="True for ternary classification (ignored if cytomine_binary is True).")
    p.add_option("--cytomine_group1", type="string", dest="cytomine_group1", help="The terms to map in the first class (ignored if cytomine_ternary is False or ignored)")
    p.add_option("--cytomine_group2", type="string", dest="cytomine_group2", help="The terms to map in the second class (ignored if cytomine_ternary is False or ignored).")
    p.add_option("--cytomine_group3", type="string", dest="cytomine_group3", help="The terms to map in the third class (ignored if cytomine_ternary is False or ignored).")

    # Images in the test set ?
    p.add_option("--cytomine_test_images", type="string", dest="cytomine_test_images", help="Some images to place in the test set for final model evaluation.")

    # Method parameters
    # Extra-trees
    p.add_option("--forest_n_estimators", type="int", default=10, dest="forest_n_estimators", help="The number of tress in pyxit underlying forest.")
    p.add_option("--forest_min_samples_split", type="string", default="1", dest="forest_min_samples_split", help="The minimum number of objects in a node for splitting (can be tuned).")
    p.add_option("--forest_max_features", type="string", default="16", dest="forest_max_features", help="The maximum number of attribute in which to look for a split when expending a node (can be tuned).")

    # Pyxit
    p.add_option("--pyxit_tune_by_cv", type="string", default="False", dest="pyxit_tune_by_cv", help="True for tuning the hyperparameters of pyxit.")
    p.add_option("--pyxit_target_width", type="int", default=16, dest="pyxit_target_width", help="Target width for the pyxit algorithm extracted windows.")
    p.add_option("--pyxit_target_height", type="int", default=16, dest="pyxit_target_height", help="Target height for the pyxit algorithm extracted windows.")
    p.add_option("--pyxit_fixed_size", type="string", default="False", dest="pyxit_fixed_size", help="True for extracting windows having a fixed size, False for randomly picked size.")
    p.add_option("--pyxit_n_subwindows", type="int", default=10, dest="pyxit_n_subwindows", help="Number of subwindows to extract per image.")
    p.add_option("--pyxit_transpose", type="string", default="False", dest="pyxit_transpose", help="True for applying rotation to the windows, False otherwise.")
    p.add_option("--pyxit_interpolation", type="int", default=2, dest="pyxit_interpolation", help="Interpolation to use (1 for nearest, 2 for bilinear, 3 for cubic and 4 for anti-alias) (can be tuned).")
    p.add_option("--pyxit_colorspace", type="string", default="2", dest="pyxit_colorspace", help="Color space the windows are converted into (0 for RGB, 1 for TRGB, 2 for HSV, 3 for GRAY) (can be tuned).")
    p.add_option("--pyxit_max_size", type="string", default="0.1", dest="pyxit_max_size", help="Maximum size proportion of the windows to extract (relative to the full image size) (can be tuned).")
    p.add_option("--pyxit_min_size", type="string", default="0.9", dest="pyxit_min_size", help="Minimum size proportion of the windows to extract (relative to the full image size) (can be tuned).")

    # Using ET-FL instead of ET-DIC
    p.add_option("--svm", type="string", default="False", dest="svm", help="True for using the ET-FL variant of Pyxit")
    p.add_option("--svm_c", type="string", default="1.0", dest="svm_c", help="SVM C parameter (can be tuned).")

    # Miscellaneous
    p.add_option("--cv_images_out", type="int", default=1, dest="cv_images_out", help="The number of images to leave out for the cross validation")
    p.add_option("--pyxit_dir_ls", type="string", default="/tmp/ls", dest="pyxit_dir_ls", help="The directory in which will be stored the images of the learning set.")
    p.add_option("--n_jobs", type="int", default=1, dest="n_jobs", help="Number of jobs for performing the cross validation.")
    p.add_option("--pyxit_save_to", type="string", default=None, dest="pyxit_save_to", help="The file path to which the best model should be saved (by default the model is not saved.")
    p.add_option("--verbose", type="string", default="False", dest="verbose", help="True for enabling verbosity.")

    params, arguments = p.parse_args(args=argv)
    params.cytomine_include_reviewed = str2bool(params.cytomine_include_reviewed)
    params.cytomine_binary = str2bool(params.cytomine_binary)
    params.cytomine_ternary = str2bool(params.cytomine_ternary)
    params.verbose = str2bool(params.verbose)
    params.pyxit_tune_by_cv = str2bool(params.pyxit_tune_by_cv)
    params.pyxit_fixed_size = str2bool(params.pyxit_fixed_size)
    params.pyxit_transpose = str2bool(params.pyxit_transpose)
    params.svm = str2bool(params.svm)

    params.cytomine_reviewed_users = str2list(params.cytomine_reviewed_users)
    params.cytomine_reviewed_images = str2list(params.cytomine_reviewed_images)
    params.cytomine_excluded_terms = str2list(params.cytomine_excluded_terms)
    params.cytomine_excluded_annotations = str2list(params.cytomine_excluded_annotations)
    params.cytomine_excluded_images = str2list(params.cytomine_excluded_images)
    params.cytomine_selected_users = str2list(params.cytomine_selected_users)
    params.cytomine_positive_terms = str2list(params.cytomine_positive_terms)
    params.cytomine_negative_terms = str2list(params.cytomine_negative_terms)
    params.cytomine_group1 = str2list(params.cytomine_group1)
    params.cytomine_group2 = str2list(params.cytomine_group2)
    params.cytomine_group3 = str2list(params.cytomine_group3)
    params.cytomine_test_images = str2list(params.cytomine_test_images)
    params.forest_min_samples_split = str2list(params.forest_min_samples_split)
    params.forest_max_features = str2list(params.forest_max_features)
    params.pyxit_colorspace = str2list(params.pyxit_colorspace)
    params.pyxit_max_size = str2list(params.pyxit_max_size, conv=float)
    params.pyxit_min_size = str2list(params.pyxit_min_size, conv=float)
    params.svm_c = str2list(params.svm_c, conv=float)

    params.pyxit_save_to = default(params.pyxit_save_to, None)
    params.pyxit_dir_ls = default(params.pyxit_dir_ls, "/tmp/ls")

    logger = Logger(params.verbose)
    logger.log("Parameters : {}".format(params))

    # Create pyxit and generate dataset
    if params.svm:
        logger.log("SVM enabled!")

    logger.log("Create Pyxit...")
    pyxit = mk_pyxit(params)

    logger.log(os.linesep + "Create dataset...")
    X, y, labels = mk_dataset(params, logger)

    logger.log(os.linesep + "Compute window sizes...")
    windows_sizes = mk_window_size_tuples(params.pyxit_min_size, params.pyxit_max_size)

    # prepare test set if needed
    is_test_set_provided = len(params.cytomine_test_images) > 0
    X_test, y_test, labels_test = [], [], []
    if is_test_set_provided:
        logger.log(os.linesep + "Test images provided. Perform train/test split...")
        X, y, labels, X_test, y_test, labels_test = train_test_split(X, y, labels, params.cytomine_test_images)
        logger.log("Annotations in test set : {}".format(X_test.shape[0]))

    logger.log(os.linesep + "Dataset size {}".format("(considering the test set)" if is_test_set_provided else ""))
    logger.log("- X     : {}".format(X.shape))
    logger.log("- y     : {}".format(y.shape))
    logger.log("- labels: {}".format(labels.shape))
    if is_test_set_provided:
        logger.log("- X_test      : {}".format(X_test.shape))
        logger.log("- y_test      : {}".format(y_test.shape))
        logger.log("- labels_test : {}".format(labels_test.shape))

    X, y, labels = np.array(X), np.array(y), np.array(labels)

    # transform into a binary/ternary problem if necessary
    is_mapped = params.cytomine_binary or params.cytomine_ternary
    mapping_dict = dict()
    if is_mapped:
        if params.cytomine_binary:
            logger.log(os.linesep + "Transform into a binary problem")
            mapper = BinaryMapper(params.cytomine_positive_terms, params.cytomine_negative_terms)
        else:
            logger.log(os.linesep + "Transform into a ternary problem")
            mapper = TernaryMapper(params.cytomine_group1, params.cytomine_group2, params.cytomine_group3)

        merged_labels = np.hstack((y, y_test)).astype(np.int64)
        mapping_dict = mapper.map_dict(merged_labels)
        y = np.array([mapper.map(to_map) for to_map in y])
        y_test = np.array([mapper.map(to_map) for to_map in y_test])

    # Print parameters
    logger.log(os.linesep + "Parameters: ")
    logger.log("- pyxit_interpolation: {}".format(params.pyxit_interpolation))
    logger.log("- pyxit_min_size : {}".format(
        params.pyxit_min_size if params.pyxit_tune_by_cv else params.pyxit_min_size[0]
    ))
    logger.log("- pyxit_max_size : {}".format(
        params.pyxit_max_size if params.pyxit_tune_by_cv else params.pyxit_max_size[0]
    ))
    logger.log("- window_size : {}".format(
        windows_sizes if params.pyxit_tune_by_cv else windows_sizes[0]
    ))
    logger.log("- forest_max_features : {}".format(
        params.forest_max_features if params.pyxit_tune_by_cv else params.forest_max_features[0]
    ))
    logger.log("- forest_min_sample_split : {}".format(
        params.forest_min_samples_split if params.pyxit_tune_by_cv else params.forest_min_samples_split[0]
    ))
    logger.log("- pyxit_colorspace : {}".format(
        params.pyxit_colorspace if params.pyxit_tune_by_cv else params.pyxit_colorspace[0]
    ))
    if params.svm:
        logger.log("- svm_c : {}".format(
            params.svm_c if params.pyxit_tune_by_cv else params.svm_c[0]
        ))

    # simply train or tune by cb ?
    if params.pyxit_tune_by_cv:
        # prepare the cross validation grid
        logger.log(os.linesep + "Prepare cross validation grid...")

        cv_params = {
            "window_sizes": windows_sizes,
            "max_features": params.forest_max_features,
            "min_samples_split": params.forest_min_samples_split,
            "colorspace": params.pyxit_colorspace
        }

        if params.svm:
            cv_params["C"] = params.svm_c

        grid = GridSearchCV(
            pyxit, cv_params,
            scoring=accuracy_scoring,
            cv=LeavePGroupsOut(params.cv_images_out),
            verbose=10,
            n_jobs=1
        )

        grid.fit(X, y, labels)

        # Extract best parameters
        best_params = grid.best_params_
        best_score = grid.best_score_
        pyxit = grid.best_estimator_  # so that pyxit has the best parameters for training the last model

        logger.log(os.linesep + "Best parameters : {}".format(best_params))
        logger.log("Best score      : {}".format(best_score))
        logger.log(os.linesep + "Re-train pyxit with best parameters...")
    else:
        logger.log(os.linesep + "Train pyxit...")

    # train the final estimator
    final_estimator = pyxit.fit(X, y)

    # save model if requested
    if params.pyxit_save_to is not None:
        logger.log(os.linesep + "Save model at {}...".format(params.pyxit_save_to))
        path = params.pyxit_save_to
        # create directory if it doesn"t exist
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        # pickle the model
        with open(path, "w+") as file:
            pickle_pyxit_adapter(final_estimator, file)

    if is_test_set_provided:
        logger.log(os.linesep + "Evaluate performances on test set...")
        logger.log("Extract subwindows on test set...")
        _X_test, _y_test = final_estimator.extract_subwindows(X_test, y_test)
        logger.log("Apply best model on test set...")
        y_pred = final_estimator.predict(X_test, _X=_X_test)
        probas = final_estimator.predict_proba(X_test, _X=_X_test)
        cm = confusion_matrix(y_test, y_pred)

        # display class correspondance
        if is_mapped:
            logger.log("Class correspondances:")
            for key in mapping_dict:
                logger.log(" - {}: {}".format(key, mapping_dict[key]))

        # display scores
        logger.log("Score(s) :")
        classes = np.union1d(np.unique(y_test).tolist(), np.unique(y).tolist()).astype("string")
        logger.log(" - Accuracy : {}".format(accuracy_score(y_test, y_pred)))
        if classes.shape[0] == 2:
            logger.log(" - Precision : {}".format(precision_score(y_test, y_pred)))
            logger.log(" - Recall    : {}".format(recall_score(y_test, y_pred)))
            logger.log(" - ROC AUC   : {}".format(roc_auc_score(y_test, probas[:, 1])))

        logger.log("Confusion matrix: ")
        print_cm(cm, classes)

if __name__ == "__main__":
    argv = [sys.argv[0],
             "--cytomine_host", "beta.cytomine.be",
             "--cytomine_public_key", "ad014190-2fba-45de-a09f-8665f803ee0b",
             "--cytomine_private_key", "767512dd-e66f-4d3c-bb46-306fa413a5eb",
             "--cytomine_base_path", "/api/",
             "--cytomine_working_path", "~/data/thyroid/wpath",
             "--cytomine_zoom_level", "0",
             "--cytomine_id_project", "716498",
             "--cytomine_id_software", "179703916",
             "--cytomine_users", "671279",
             "--cytomine_excluded_terms", "675999,676026,933004,8844862,8844845,9444456,15054705,15054765,15109451,15109495,22042230,28792193,30559888",
             "--cytomine_excluded_annotations", "30675573,18107252,9321884,7994253,9313842",
             "--cytomine_excluded_images", "",
             "--cytomine_include_reviewed", "False",
             "--cytomine_reviewed_users", "14",
             "--cytomine_reviewed_images", "8120444",
             "--cytomine_binary", "True",
             "--cytomine_positive_terms", "676390",
             "--cytomine_negative_terms", "676446,676210,676434,676176,676407,15109483,15109489",
             "--cytomine_ternary", "False",
             "--cytomine_group1", "",
             "--cytomine_group2", "",
             "--cytomine_group3", "",
             "--cytomine_test_images", "8124112,8123867,8122868,8122830,8120497,8120408,8120321,728799,728744,728725,728709,728689,728675,728391,724858,719625,716534,716528",
             "--forest_n_estimators", "10",
             "--forest_min_samples_split", "156",
             "--forest_max_features", "1,28,384,768",
             "--pyxit_tune_by_cv", "True",
             "--pyxit_target_width", "16",
             "--pyxit_target_height", "16",
             "--pyxit_fixed_size", "True",
             "--pyxit_n_subwindows", "1",
             "--pyxit_transpose", "False",
             "--pyxit_interpolation", "2",
             "--pyxit_colorspace", "2",
             "--pyxit_min_size", "0.1",
             "--pyxit_max_size", "0.9",
             "--svm", "False",
             "--svm_c", "1.0",
             "--cv_images_out", "1",
             "--pyxit_dir_ls", "",
             "--n_jobs", "2",
             "--pyxit_save_to", "",
             "--verbose", "True"]
    main(argv)
