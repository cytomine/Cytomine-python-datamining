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

from abc import ABCMeta, abstractmethod

import numpy as np

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class LabelMapper(object):
    """An class for defining a mapping between a set of labels into other labels"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def map(self, label):
        """Map an input label with the expected label

        Parameters
        ----------
        label:
            The input label

        Returns
        -------
        output:
            The output label
        """
        pass

    def map_dict(self, labels):
        """
        Create a dictionary of which the keys are mapped labels and values the set
        of input labels that were converted to the mapped label

        Parameters
        ----------
        labels: list
            The labels to map

        Returns
        -------
        dict: dict
            A dictionary mapping output label with input labels
        """
        unique_terms = np.unique(np.array(labels))
        map_dict = dict()
        for term in unique_terms.tolist():
            mapped = self.map(term)
            map_dict[mapped] = map_dict.get(mapped, []) + [term]
        return map_dict


class IdMapper(LabelMapper):
    """The identity mapper"""
    def map(self, term):
        return term


class GroupMapper(LabelMapper):
    """A mapper which maps labels of groups into an output label"""
    def __init__(self, groups, output_labels=None):
        """
        Parameters
        ----------
        groups: list of list of labels
            Each sublist correspond to a groupe of labels to convert into a unique output label
        output_labels: list (optional, size: len(groups))
            The list of labels to which must be mapped each group. By default, output labels
            are integers from 0 to len(groups) - 1.
        """
        self._groups = [set(g) for g in groups]
        self._output_labels = range(0, len(groups)) if output_labels is None else output_labels

    def map(self, label):
        for i, group in enumerate(self._groups):
            if label in group:
                return self._output_labels[i]
        raise ValueError("Couldn't map unexpected label: {} ({})".format(label, type(label)))


class BinaryMapper(GroupMapper):
    """A binary mapper"""
    def __init__(self, positive_classes, negative_classes, positive_label=1, negative_label=0):
        super(BinaryMapper, self).__init__([positive_classes, negative_classes], [positive_label, negative_label])


class TernaryMapper(BinaryMapper):
    """A ternary mapper"""
    def __init__(self, first_classes, second_classes, third_classes, first_label=0, second_label=1, third_label=2):
        super(TernaryMapper, self).__init__([
            first_classes, second_classes, third_classes
        ], [
            first_label, second_label, third_label
        ])