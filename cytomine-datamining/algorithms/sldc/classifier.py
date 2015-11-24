# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__contributors__ = ["Mormont Romain <r.mormont@student.ulg.ac.be>"]
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'

import numpy as np

class Classifier(object):
    """
    """

    def __init__(self):
       pass

    def classify(self, datastore):
        pass


class ThyroidClassifier(Classifier):
    """
    =================
    ThyroidClassifier
    =================
    The :class:`Classifier` for the Thyorid cell classification project.
    It classifies cells and architectural pattern separately. The
    classification scheme (and the types of results) depends on its arguments.

    Consutrctor parameters
    ----------------------
    cell_classifier : callable(:class:ImageBuffer)
        A function which can classify the cells.
        The label semantcis depends on the classifier.
    arch_pattern_classifier : callable(:class:ImageBuffer)
        A function which can classify the architectural patterns.
        The label semantcis depends on the classifier.
    """

    def __init__(self, cell_classifier, arch_pattern_classifier, cell_classes=None, arch_pattern_classes=None):
        Classifier.__init__(self)
        self.cell_classifier = cell_classifier
        self.pattern_classifier = arch_pattern_classifier
        self._cell_classes = cell_classes
        self._arch_pattern_classes = arch_pattern_classes

    def classify(self, datastore, return_rates=False):
        """
        Performs the classification of
        - Cells into
            - presence of inclusion or pseudo-inclusion
            - absence of inclusion
        - Architectural pattern into
            - Normal
            - Proliferative
            - Artifact/background/colloid

        Parameters
        ----------
        datastore : :class:`ThyroidDataStore`
            The datastore from which to load the cells and architectural
            pattern images to classify

        Return
        -----
        tuple = (classif1, classif2)
            classif1 : cell_classifier_function dependent
                The result of the cell classification.
            classif2 : arch_pattern_classifier_function dependent
        """

        #Cells
        cells_image_stream = datastore.get_cells_to_classify()
        cell_rates = None
        if self._cell_classes is None:
            cell_results = self.cell_classifier.predict(cells_image_stream)
        else:
            cell_probas = self.cell_classifier.predict_proba(cells_image_stream)
            cell_results = self._cell_classes.take(np.argmax(cell_probas, axis=1), axis=0)
            cell_rates = np.max(cell_probas, axis=1)

        #Architectural pattern
        arch_pattern_image_stream = datastore.get_arch_pattern_to_classify()
        arch_pattern_rates = None
        if self._arch_pattern_classes is None:
            arch_pattern_results = self.pattern_classifier.predict(arch_pattern_image_stream)
        else:
            arch_pattern_probas = self.pattern_classifier.predict_proba(arch_pattern_image_stream)
            arch_pattern_results = self._arch_pattern_classes.take(np.argmax(arch_pattern_probas, axis=1), axis=0)
            arch_pattern_rates = np.max(arch_pattern_probas)

        if return_rates:
            return cell_results, arch_pattern_results, cell_rates, arch_pattern_rates
        else:
            return cell_results, arch_pattern_results
