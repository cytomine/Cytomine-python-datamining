# -*- coding: utf-8 -*-

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class ThyroidOntology(object):
    """Class for storing ids of the thyroid ontology
    """
    CELL_NOS = 676446
    CELL_INCL = 676390
    CELL_PSEUDO = 676210
    CELL_GLASS = 676434
    CELL_NORM = 676176
    CELL_GROOVES = 676407
    PATTERN_NORM = 675999
    PATTERN_PROLIF = 676026
    PATTERN_MINOR = 933004
    BACKGROUND = 8844862
    ARTIFACTS = 8844845
    MACROPHAGES = 15109451
    CELL_BLOOD = 15109483
    CELL_POLY = 15109489
    COLLOID = 15109495

    def __init__(self, onto_id):
        self._onto_id = onto_id

    def __str__(self):
        return ThyroidOntology.name(self._onto_id)

    @classmethod
    def name(cls, id):
        if id == cls.CELL_NOS:
            return "Papillary cell NOS"
        elif id == cls.CELL_INCL:
            return "Papillary cell with inclusion"
        elif id == cls.CELL_PSEUDO:
            return "Normal follicular cell with pseudo-inclusion (artefact)"
        elif id == cls.CELL_GLASS:
            return "Papillary cell with ground glass nuclei"
        elif id == cls.CELL_NORM:
            return "Normal follicular cells"
        elif id == cls.CELL_GROOVES:
            return "Papillary cell with nuclear grooves"
        elif id == cls.PATTERN_NORM:
            return "Normal follicular architectural pattern"
        elif id == cls.PATTERN_PROLIF:
            return "Proliferative follicular architectural pattern"
        elif id == cls.PATTERN_MINOR:
            return "Proliferative follicular architectural pattern (minor sign)"
        elif id == cls.BACKGROUND:
            return "Background"
        elif id == cls.ARTIFACTS:
            return "Artefacts"
        elif id == cls.MACROPHAGES:
            return "Macrophages"
        elif id == cls.CELL_BLOOD:
            return "Red blood cells"
        elif id == cls.CELL_POLY:
            return "PN (polynuclear)"
        elif id == cls.COLLOID:
            return "Colloid"
        else:
            raise ValueError("Unknown ontology id")

    @staticmethod
    def instance(onto_id):
        return ThyroidOntology(onto_id)

