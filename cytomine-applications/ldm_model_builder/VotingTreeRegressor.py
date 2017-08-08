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

import numpy as np
from sklearn.tree import ExtraTreeRegressor
from multiprocessing import Pool


def build_voting_tree_regressor(X,y,max_features,max_depth,min_samples_split):
	clf = ExtraTreeRegressor(max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split)
	clf = clf.fit(X,y)
	return clf


def votingtree_reg_training_mp_helper(jobargs):
	return build_voting_tree_regressor(*jobargs)


def votingtree_reg_test_mp_helper(jobargs):
	return test_voting_tree_reg(*jobargs)


def test_voting_tree_reg(tree,X):
	return tree.predict(X)


class VotingTreeRegressor:

	def __init__(self,n_estimators=10,max_features='auto',max_depth=None,min_samples_split=2,n_jobs=1):
		self.n_estimators = n_estimators
		self.max_features = max_features
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.n_jobs = n_jobs
		
	def fit(self,X,y):
		self.trees = []
		self.n_classes = np.max(y)+1
		(h,w) = X.shape
		p = Pool(self.n_jobs)
		jobargs = [(X,y,self.max_features,self.max_depth,self.min_samples_split) for i in range(self.n_estimators)]
		self.trees = p.map(votingtree_reg_training_mp_helper,jobargs)
		p.close()
		p.join()
		
		return self
	
	def predict(self,X):
		(h,w) = X.shape
		n_features = w/self.n_estimators	
		p = Pool(self.n_jobs)
		jobargs = [(self.trees[i],X) for i in range(self.n_estimators)]
		probas = p.map(votingtree_reg_test_mp_helper,jobargs)
		p.close()
		p.join()
		return probas
		

def test():
	tr_data = np.random.ranf((10000,3200))
	tr_rep = np.random.ranf((10000,2))
	te_data = np.random.ranf((100,3200))
	clf = VotingTreeRegressor(n_estimators=32,max_features=2,n_jobs=4)
	clf.fit(tr_data,tr_rep)
	clf.predict(te_data)[1]


if __name__ == "__main__":
	test()
