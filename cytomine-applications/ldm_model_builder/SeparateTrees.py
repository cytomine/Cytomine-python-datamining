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
from sklearn.tree import ExtraTreeClassifier
from multiprocessing import Pool

def build_separate_tree(X,y,max_features,max_depth,min_samples_split):
	clf = ExtraTreeClassifier(max_features=max_features,max_depth=max_depth,min_samples_split=min_samples_split)
	clf = clf.fit(X,y)
	return clf
	
def separatetree_training_mp_helper(jobargs):
	return build_separate_tree(*jobargs)
	
def separatetree_test_mp_helper(jobargs):
	return test_separate_tree(*jobargs)

def test_separate_tree(tree,X):
	return tree.predict_proba(X)
	
class SeparateTrees:

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
		n_features = w/self.n_estimators
		
		p = Pool(self.n_jobs)
		jobargs = [(X[:,i*n_features:(i+1)*n_features],y,self.max_features,self.max_depth,self.min_samples_split) for i in range(self.n_estimators)]
		self.trees = p.map(separatetree_training_mp_helper,jobargs)
		p.close()
		p.join()
		
		return self
	
	def predict_proba(self,X):
		(h,w) = X.shape
		n_features = w/self.n_estimators	
		p = Pool(self.n_jobs)
		jobargs = [(self.trees[i],X[:,i*n_features:(i+1)*n_features]) for i in range(self.n_estimators)]
		probas = p.map(separatetree_test_mp_helper,jobargs)
		p.close()
		p.join()
		return np.sum(probas,axis=0)/float(self.n_estimators)

	def predict(self,X):
		probas = self.predict_proba(X)
		return np.argmax(probas,axis=1)
		

if __name__ == "__main__":
	clf = SeparateTrees(n_estimators=32,max_features=2,n_jobs=4)
	clf.fit(np.random.ranf((10000,3200)),np.random.randint(0,2,10000))
	print clf.predict_proba(np.random.ranf((100,3200)))
	print clf.predict(np.random.ranf((100,3200)))
