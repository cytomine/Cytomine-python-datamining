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

__author__          = "Gilles Louppe"
__contributors__    = ["Marée Raphaël <raphael.maree@ulg.ac.be>", "Stévens Benjamin <b.stevens@ulg.ac.be>"]
__copyright__       = "Copyright 2010-2015 University of Liège, Belgium, http://www.cytomine.be/"




import cPickle as pickle
import numpy as np
import sys

from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import neighbors

from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from data import build_from_dir
from estimator import PyxitClassifier, MAX_INT

import optparse

#CONSTANT to define final classifier using ET-FL features (Liblinear recommended)
SVM_LIBSVM=1
SVM_LIBLINEAR=2
SVM_LRL1=3
SVM_LRL2=4
ET=5
RF=6
NN=7


#Enhanced display of confusion matrix
def print_cm(cm, classes):
    totalsamples=0
    classERSum=0
    totalrecognized=0
    print "%-10s" % "",
    for c in classes:
	c=str(c)
        print "%-10s" % c[:min(10,len(c))],
    print

    for i in xrange(cm.shape[0]):
        print "%-10s" % classes[i], #[:min(10, len(str(classes[i])))],                                                                                                         

        totalrow=0
        for j in xrange(cm.shape[1]):
            print "%-10d" % cm[i, j],
            totalrow+=cm[i,j]

        classER=100*(float(cm[i,i])/float(totalrow))
        classERSum+=classER
        totalrecognized+=cm[i,i]
        totalsamples+=totalrow
        print "%d / %d = %.2f %%" %(cm[i,i],totalrow,classER)
        print
    print "Totalrecognized:%d , Totalsamples: %d" %(totalrecognized,totalsamples)
    print "Average class recognition rate: %.2f %%" %(classERSum/len(classes))
    print "Overall recognition rate: %.2f %%" %(100*float(totalrecognized)/totalsamples)



#Simplified display of Confusion Matrix
def print_cm_simplified(cm, classes):
    print "%-10s" % "",
    for c in classes:
        c=str(c)
        print "%-10s" % c[:min(10,len(c))],
    print

    for i in xrange(cm.shape[0]):
        print "%-10s" % classes[i], #[:min(10, len(str(classes[i])))],

        for j in xrange(cm.shape[1]):
            print "%-10d" % cm[i, j],

        print


def main(argv):
    # Define command line options
    p = optparse.OptionParser(description='Pyxit',
                              prog='PyXit (PYthon piXiT)',
                              version='PyXit 0.1')

    p.add_option('--dir_ls', type="string", dest="dir_ls", help="The learning set directory")
    p.add_option('--dir_ts', type="string", dest="dir_ts", help="The training set directory")

    p.add_option('--cv_k_folds', type="int", dest="cv_k_folds", help="The number of folds")
    p.add_option('--cv_shuffle', default=False, action="store_true", dest="cv_shuffle", help="Whether cross-validation is performed using ShuffleSplit.")
    p.add_option('--cv_shuffle_test_fraction', default=0.1, type="float", dest="cv_shuffle_test_fraction", help="The proportion of data in shuffled test splits.")

    p.add_option('--pyxit_n_subwindows', default=10, type="int", dest="pyxit_n_subwindows", help="number of subwindows")
    p.add_option('--pyxit_min_size', default=0.5, type="float", dest="pyxit_min_size", help="min size")
    p.add_option('--pyxit_max_size', default=1.0, type="float", dest="pyxit_max_size", help="max size")
    p.add_option('--pyxit_target_width', default=16, type="int", dest="pyxit_target_width",  help="target width")
    p.add_option('--pyxit_target_height', default=16, type="int", dest="pyxit_target_height", help="target height")
    p.add_option('--pyxit_interpolation', default=2, type="int", dest="pyxit_interpolation", help="interpolation method 1,2,3,4")
    p.add_option('--pyxit_transpose', default=False, action="store_true", dest="pyxit_transpose", help="transpose subwindows")
    p.add_option('--pyxit_colorspace', default=2, type="int", dest="pyxit_colorspace", help="colorspace 0=RGB, 1=TRGB, 2=HSV")
    p.add_option('--pyxit_fixed_size', default=False, action="store_true", dest="pyxit_fixed_size", help="extract fixed size subwindows")
    p.add_option('--pyxit_n_jobs', default=1, type="int", dest="pyxit_n_jobs", help="number of jobs")
    p.add_option('--pyxit_save_to', type="string", dest="pyxit_save_to", help="file to save the model into")

    p.add_option('--forest_n_estimators', default=10, type="int", dest="forest_n_estimators", help="number of base estimators (T)")
    p.add_option('--forest_max_features' , default=1, type="int", dest="forest_max_features", help="max features at test node (k)")
    p.add_option('--forest_min_samples_split', default=1, type="int", dest="forest_min_samples_split", help="minimum node sample size (nmin)")
    p.add_option('--forest_shared_mem', default=False, action="store_true", dest="forest_shared_mem", help="shared mem")

    p.add_option('--svm', default=0, dest="svm", help="final svm classifier: 0=nosvm, 1=libsvm, 2=liblinear, 3=lr-l1, 4=lr-l2", type="int")
    p.add_option('--svm_c', default=1.0, type="float", dest="svm_c", help="svm C")

    p.add_option('--quiet', action="store_false", default=True, dest="verbose", help="Turn off verbose mode")
    p.add_option('--verbose', action="store_true", default=True, dest="verbose", help="Turn on verbose mode")

    options, arguments = p.parse_args( args = argv)

    # Check for errors in the options
    e = None

    if not options.dir_ls:
        e = "--dir_ls needs to be set."

    elif options.dir_ts and options.cv_k_folds:
        e = "--dir_ts and --cv_k_folds cannot be set at the same time."

    elif options.pyxit_save_to and options.cv_k_folds:
        e = "--pyxit_save_to and --cv_k_folds cannot be set at the time."

    if e:
        print "Error: %s" % e
        print "Run with -h option for help."
        sys.exit(1)

    if options.verbose:
      print "[pyxit.main] Options = ", options


    # Load data
    if options.verbose:
        print "[pyxit.main] Loading data..."

    X, y = build_from_dir(options.dir_ls)

    classes = np.unique(y)
    n_classes = len(classes)
    y_original = y
    y = np.searchsorted(classes, y)

    # Instantiate classifiers
    if options.verbose:
        print "[pyxit.main] Initializing PyxitClassifier..."

    forest = ExtraTreesClassifier(n_estimators=options.forest_n_estimators,
                                  max_features=options.forest_max_features,
                                  min_samples_split=options.forest_min_samples_split,
                                  n_jobs=options.pyxit_n_jobs,
                                  verbose=options.verbose)

    pyxit = PyxitClassifier(base_estimator=forest,
                            n_subwindows=options.pyxit_n_subwindows,
                            min_size=options.pyxit_min_size,
                            max_size=options.pyxit_max_size,
                            target_width=options.pyxit_target_width,
                            target_height=options.pyxit_target_height,
                            interpolation=options.pyxit_interpolation,
                            transpose=options.pyxit_transpose,
                            colorspace=options.pyxit_colorspace,
                            fixed_size=options.pyxit_fixed_size,
                            n_jobs=options.pyxit_n_jobs,
                            verbose=options.verbose)

    if options.svm:
        if options.svm == SVM_LIBSVM:
            svm = SVC(probability=True, C=options.svm_c, kernel="linear")
        if options.svm == SVM_LIBLINEAR:
            svm = LinearSVC(C=options.svm_c)
        if options.svm == SVM_LRL1:
            svm = LogisticRegression(penalty="l1", C=options.svm_c)
        if options.svm == SVM_LRL2:
            svm = LogisticRegression(penalty="l2", C=options.svm_c)
        if options.svm == ET:
            svm = ExtraTreesClassifier(n_estimators=1000,
                                       max_features="sqrt",
                                       #max_features=1000,
                                       min_samples_split=2,
                                       n_jobs=options.pyxit_n_jobs,
                                       verbose=options.verbose)
        if options.svm == RF:
            svm = RandomForestClassifier(n_estimators=1000,
                                         #max_features=1000,
                                         max_features="sqrt",
                                         min_samples_split=2,
                                         n_jobs=options.pyxit_n_jobs,
                                         verbose=options.verbose)

        if options.svm == NN:
            svm = neighbors.KNeighborsClassifier(10) 



    if options.verbose:
        print "[pyxit.main] PyxitClassifier ="
        print pyxit

        if options.svm:
            print "[pyxit.main] SVM ="
            print svm

    # Build and evaluate
    if options.dir_ls and not options.dir_ts and not options.cv_k_folds:
        if options.pyxit_save_to:
            fd = open(options.pyxit_save_to, "wb")
            pickle.dump(classes, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.verbose:
            print "[pyxit.main] Fitting PyxitClassifier on %s" % options.dir_ls

        _X, _y = pyxit.extract_subwindows(X, y)
        pyxit.fit(X, y, _X=_X, _y=_y)

        if options.verbose:
            print "[pyxit.main] Saving PyxitClassifier into %s" % options.pyxit_save_to

        if options.pyxit_save_to:
            pickle.dump(pyxit, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.svm:
            Xt = pyxit.transform(X, _X=_X, reset=True)

            if options.verbose:
                print "[pyxit.main] Fitting SVC on %s" % options.dir_ls

            svm.fit(Xt, y)

            if options.verbose:
                print "[pyxit.main] Saving SVM into %s" % options.pyxit_save_to

            if options.pyxit_save_to:
                pickle.dump(svm, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.pyxit_save_to:
            fd.close()

    elif options.dir_ts:
        if options.pyxit_save_to:
            fd = open(options.pyxit_save_to, "wb")
            pickle.dump(classes, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.verbose:
            print "[pyxit.main] Fitting PyxitClassifier on %s" % options.dir_ls

        _X, _y = pyxit.extract_subwindows(X, y)
        pyxit.fit(X, y, _X=_X, _y=_y)

        if options.pyxit_save_to:
            pickle.dump(pyxit, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.svm:
            Xt = pyxit.transform(X, _X=_X, reset=True)

            if options.verbose:
                print "[pyxit.main] Fitting SVC on %s" % options.dir_ls

            svm.fit(Xt, y)

            if options.pyxit_save_to:
                pickle.dump(svm, fd, protocol=pickle.HIGHEST_PROTOCOL)

        if options.pyxit_save_to:
            fd.close()

        if options.verbose:
            print "[pyxit.main] Testing on %s" % options.dir_ts

        X_test, y_test = build_from_dir(options.dir_ts)
        y_test = np.searchsorted(classes, y_test)
        _X_test, _y_test = pyxit.extract_subwindows(X_test, y_test)
        y_true = y_test
        all_tested = np.ones(len(y_true), dtype=np.bool)

        if not options.svm:
            y_predict = pyxit.predict(X_test, _X=_X_test)
            y_proba = pyxit.predict_proba(X_test, _X=_X_test)

        else:
            Xt = pyxit.transform(X_test, _X=_X_test)
            y_predict = svm.predict(Xt)
            if options.svm!=SVM_LIBLINEAR:
                y_proba = svm.predict_proba(Xt)

    elif options.cv_k_folds:
        if options.verbose:
            print "[pyxit.main] K-Fold cross-validation (K=%d)" % options.cv_k_folds

        _X, _y = pyxit.extract_subwindows(X, y)

        i = 1
        step = 100. / options.cv_k_folds

        y_true = y
        y_predict = np.empty(y_true.shape, dtype=y.dtype)
        y_proba = np.empty((y_true.shape[0], n_classes))
        all_tested = np.zeros(len(y_true), dtype=np.bool)

        cm = np.zeros((n_classes, n_classes), dtype=np.int32)

        if not options.cv_shuffle:
            cv = StratifiedKFold(y_true, options.cv_k_folds)
        else:
            cv = ShuffleSplit(len(X), n_iter=options.cv_k_folds, test_size=options.cv_shuffle_test_fraction)

        for train, test in cv:
            all_tested[test] = True
            _train = pyxit.extend_mask(train)
            _test = pyxit.extend_mask(test)

            if options.verbose:
                print "[pyxit.main] Fitting PyxitClassifier on fold %d" % i

            pyxit.fit(X[train], y[train], _X=_X[_train], _y=_y[_train])

            if options.svm:
                Xt = pyxit.transform(X[train], _X=_X[_train], reset=True)

                if options.verbose:
                    print "[pyxit.main] Fitting SVC on fold %d" % i

                svm.fit(Xt, y[train])

            if options.verbose:
                print "[pyxit.main] Testing on fold %d" % i

            if not options.svm:
                y_predict[test] = pyxit.predict(X[test], _X=_X[_test])
                y_proba[test] = pyxit.predict_proba(X[test], _X=_X[_test])

            else:
                Xt = pyxit.transform(X[test], _X=_X[_test])
                y_predict[test] = np.asarray(svm.predict(Xt), dtype=y.dtype)

                if hasattr(svm, "predict_proba"):
                    y_proba[test] = svm.predict_proba(Xt)
                print svm

            if options.verbose:
                print "[pyxit.main] Classification error on fold %d = %f" % (i, 1.0 * np.sum(y_true[test] != y_predict[test]) / len(y_true[test]))
                print "[pyxit.main] Cumulated confusion matrix ="
                cm += confusion_matrix(y_true[test], y_predict[test])
                print_cm(cm, classes)

            i += 1

    # Output some results
    if "all_tested" in locals():
        if options.verbose:
            print "---"
            print "[pyxit.main] Test coverage =", sum(all_tested) / (1.0 * len(all_tested))
            print "[pyxit.main] Overall classification error = %f" % (1.0 * np.sum(y_true[all_tested] != y_predict[all_tested]) / len(y_true[all_tested]))
            print "[pyxit.main] Overall confusion matrix ="
            print_cm(confusion_matrix(y_true[all_tested], y_predict[all_tested]), classes)

        #y_true = classes.take(y_true[all_tested], axis=0)
        y_predict = classes.take(y_predict[all_tested], axis=0)
        y_proba = np.max(y_proba, axis=1)
        d = {}
        for i in xrange(len(X)):
            d[X[i]] = (int(y_predict[i]), y_proba[i]) 
        return d

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])

