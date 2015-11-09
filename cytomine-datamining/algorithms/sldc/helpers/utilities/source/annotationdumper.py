# -*- coding: utf-8 -*-
"""
Copyright 2010-2013 University of Liège, Belgium.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the
use of this software.

Permission is only granted to use this software for non-commercial purposes.
"""

__author__ = "Begon Jean-Michel <jm.begon@gmail.com>"
__copyright__ = "Copyright 2010-2013 University of Liège, Belgium"
__version__ = '0.1'


import os
import shutil

from cytomine.models import Annotation

from .learningsetloader import LearningSetFromDir
from .imagebuffer import NotCMapPILLoader

class AnnotationDumper:
    """
    ================
    AnnotationDumper
    ================
    A :class:`AnnotationDumper` dumps annotations from Cytomine on a given
    directory. The annotations are linked to a given project at a certain
    zoom level. Some labels may be excluded

    Directory layout
    ---------------
    The directory will be filled with subdirectories. 
    Each such subdirectory corresponds to an identifier of a non-excluded term.
    Each subdirectory is filled by images of the corresponding class

    Class constant
    --------------
    CROP_ONLY : indicates that only the crop should be downloaded
    CROP_AND_MASK : indicates that both th crop and the mask should be 
        downloaded

    Usage
    -----
    This class can be used in two ways:
    1. Persistent storage
        If you want to get a persistent storage of the images, use the
        meth:`dump`method
    2. Non-persistent storage
        If you want to remove all the files created you can either use the
        meth:`dump`/meth:`close`combination or the with statement

    Constructor parameters
    ----------------------
    cytomine : :class:`Cytomine`
        The instance of the Cytomine client to interact with
    id_project : int 
        The id of the project from which to downloaded crops
    dump_dir : directory path
        The path to the directory in which to dump the images.
        If the directory does not exist, it will be created (and deleted
            afterwards if the "with" statement is used)
    zoom : int >= 0
        The zoom level at which the crops must be downloaded
    excluded_terms : list of int
        The annotation terms to exclude
    dump_type : either CROP_ONLY or CROP_AND_MASK
        Whether to include or not the mask
    """

    CROP_ONLY = 1
    CROP_AND_ALPHA = 2
    MASK = 3

    def __init__(self, cytomine, id_project, dump_dir, 
                 zoom, exluced_terms=[], dump_type=CROP_ONLY,
                 override=False):
        self._cytomine = cytomine
        self._id = id_project
        self._type = dump_type
        self._terms = exluced_terms
        self._dump_dir = dump_dir
        self._zoom = zoom
        self.override = override
        self._annotations = None
        self._remove_dir = False
        self._existing_dir = {}

    def dump(self):
        """
        dump the crop on the file system

        Return
        ------
        annotations : :class:`Annotation`
            The annotations whose crop have been dumped on the FS
        """
        if not os.path.exists(self._dump_dir):
            os.makedirs(self._dump_dir)
            self._remove_dir = True
        else:
            for directory in os.listdir(self._dump_dir):
                self._existing_dir[directory] = True

        annotations = self._cytomine.get_annotations(id_project=self._id)
        if self._type == AnnotationDumper.CROP_AND_ALPHA:
            annotation_get_func = Annotation.get_annotation_alpha_crop_url
        elif self._type == AnnotationDumper.MASK:
            annotation_get_func = Annotation.get_annotation_mask_url
        else:
            annotation_get_func = Annotation.get_annotation_crop_url
        annotations = self._cytomine.dump_annotations(annotations=annotations, 
                                                      get_image_url_func=annotation_get_func, 
                                                      dest_path=self._dump_dir, 
                                                      desired_zoom=self._zoom,
                                                      excluded_terms=self._terms,
                                                      override=self.override)
        self._annotations = annotations
        return annotations

    def get_annotations(self):
        """
        Return
        ------
        annotations : :class:`Annotation`
            The annotations whose crop have been dumped on the FS
            or None if nothing has been dumped yet
        """
        return self._annotations

    def close(self):
        """
        Restore the file system as it was before the dumping
        """
        if self._remove_dir:
            if os.path.exists(self._dump_dir):
                shutil.rmtree(self._dump_dir)
        else:
            for directory in os.listdir(self._dump_dir):
                if directory not in self._existing_dir:
                    shutil.rmtree(directory)


    def get_learningset(self, image_loader=NotCMapPILLoader(), map_classes=None):
        """
        Return a learning set corresponding to the downloaded images.

        Parameters
        ----------
        image_loader : :class:`ImageLoader`
            An :class:`ImageLoader` instance which can work with the seeds given 
        map_classes : dictionary str->int (default : None)
            An optional mapping dictionary which translates subdirectory names
            to label
        
        Return
        ------
        ls : :class:`LearningSetBuffer`
            The learning set corresponding to the downloaded images
        """
        return LearningSetFromDir(self._dump_dir, image_loader, map_classes)
                
        
    def __enter__(self):
        self.dump()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

