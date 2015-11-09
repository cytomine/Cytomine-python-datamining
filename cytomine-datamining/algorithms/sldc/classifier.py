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

    def __init__(self, cell_classifier, arch_pattern_classifier):
        Classifier.__init__(self)
        self.cell_classifier = cell_classifier
        self.pattern_classifier = arch_pattern_classifier

    def classify(self, datastore):
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
        cell_results = self.cell_classifier.predict(cells_image_stream)
        #Architectural pattern
        arch_pattern_image_stream = datastore.get_arch_pattern_to_classify()
        arch_pattern_results = self.pattern_classifier.predict(arch_pattern_image_stream)

        return cell_results, arch_pattern_results
