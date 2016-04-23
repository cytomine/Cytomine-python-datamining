# -*- coding: utf-8 -*-
import locale

import cytomine
import numpy as np
from shapely.wkt import loads

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

params = {
    'verbose' : False,
    'cytomine_host' : "beta.cytomine.be",
    'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cytomine_base_path' : '/api/',
    'cytomine_working_path' : '/home/mass/GRD/r.mormont/tmp/check',
    'project': 716498,  # id of the project from which the annotations must be fetched
    'slides': [8124821, 8124538, 8124112, 8124013, 8123992, 8123867, 8123768, 8123591, 8123101, 8122868, 8122830,
               8122730, 8122696, 8122590, 8122518, 8120497, 8120472, 8120444, 8120434, 8120424, 8120416, 8120408,
               8120400, 8120392, 8120382, 8120370, 8120351, 8120343, 8120331, 8120321, 8120309, 8120281, 8120272,
               8120257, 749384, 728799, 728791, 728783, 728772, 728755, 728744, 728733, 728725, 728717, 728709, 728689,
               728675, 728667, 728581, 728391, 728066, 727250, 724858, 723911, 722807, 720965, 719660, 719625, 716583,
               716547, 716534, 716528, 716522, 716516],  # the slides that must be checked
    'users': [671279],  # ids of the users of which the annotation must be used as ground truth
}

thyroid_ontology = {
    676446: "Papillary cell NOS",
    676390: "Papillary cell with inclusion",
    676210: "Normal follicular cell with pseudo-inclusion (artefact)",
    676434: "Papillary cell with ground glass nuclei",
    675999: "Normal follicular architectural pattern",
    676026: "Proliferative follicular architectural pattern",
    676176: "Normal follicular cells",
    933004: "Proliferative follicular architectural pattern (minor sign)",
    676407: "Papillary cell with nuclear grooves",
    8844862: "Background",
    8844845: "Artefacts",
    9444456: "To classify",
    15054705: "Architectural pattern to classify",
    15054765: "Cell to classify",
    15109451: "Macrophages",
    15109483: "Red blood cells",
    15109489: "PN (polynuclear)",
    15109495: "Colloid",
    22042230: "Region",
    28792193: "Aggregate",
    30559888: "NonInclusion"
}


class ImageCache(object):

    def __init__(self, conn):
        self._conn = conn
        self._cache = dict()

    def get_image(self, id):
        if id not in self._cache:
            self._cache[id] = self._conn.get_image_instance(id)
        return self._cache[id]


def summary(array):
    hasItems = len(array) > 0
    return {
        "min" : round(np.min(array), 5) if hasItems else -1,
        "max" : round(np.max(array), 5) if hasItems else -1,
        "std" : round(np.std(array), 5) if hasItems else -1,
        "mean": round(np.mean(array), 5) if hasItems else -1
    }


def cpu_stats(stats):
    area = np.array(stats["area"])
    area_real = np.array(stats["area_real"])
    circ = np.array(stats["circ"])
    return summary(area), summary(area_real), summary(circ)


def print_stats(id, stats):
    # print "==========================="
    # print "Term #{}".format(id if id != -1 else "NONE")
    # print "Stats : "
    area, area_real, circ = cpu_stats(stats)
    # print "  - area : {}".format(area)
    # print "  - circ : {}".format(circ)
    print "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
        id if id != -1 else "NONE",
        thyroid_ontology.get(id, "NONE"),
        len(stats["area"]),
        area["min"], area["max"], area["mean"], area["std"],
        area_real["min"], area_real["max"], area_real["mean"], area_real["std"],
        circ["min"], circ["max"], circ["mean"], circ["std"])


def circularity(polygon):
    return 4 * np.pi * polygon.area / (polygon.length * polygon.length)


def compactness(polygon):
    return polygon.area / polygon.convex_hull.area


def append(dict, id, area, area_real, circ):
    stats = dict.get(id, { "area": [], "circ": [], "area_real": [] })
    stats["area"].append(area)
    stats["area_real"].append(area_real)
    stats["circ"].append(circ)
    dict[id] = stats


def make_stats(annotations):
    stats_dict = dict()
    image_cache = ImageCache(conn)
    for i, annotation in enumerate(annotations):
        image = image_cache.get_image(annotation.image)
        polygon = loads(annotation.location)
        key = -1 if len(annotation.term) == 0 else annotation.term[0]
        append(stats_dict, key, polygon.area, polygon.area * (image.resolution * image.resolution), circularity(polygon))

    for key in stats_dict.keys():
        print_stats(key, stats_dict[key])


def make_db(annotations):
    print "annot;term;area;real_area;circ;comp"
    locale.setlocale(locale.LC_ALL, "fr_BE.UTF-8")
    format_fn = lambda v: locale.format("%.4f", v)
    image_cache = ImageCache(conn)
    for annotation in annotations:
        image = image_cache.get_image(annotation.image)
        polygon = loads(annotation.location)
        key = -1 if len(annotation.term) == 0 else annotation.term[0]
        area = format_fn(polygon.area)
        real_area = format_fn(polygon.area * (image.resolution * image.resolution))
        circ = format_fn(circularity(polygon))
        comp = format_fn(compactness(polygon))
        print "{};{};{};{};{};{}".format(annotation.id, key, area, real_area, circ, comp)

if __name__ == "__main__":
    conn = cytomine.Cytomine(params["cytomine_host"], params["cytomine_public_key"],
                             params["cytomine_private_key"], base_path = params['cytomine_base_path'],
                             working_path = params['cytomine_working_path'], verbose=params['verbose'])
    # download jobs annotation
    annotations = conn.get_annotations(id_project=params['project'], id_user=params['users'],
                                       id_image=params['slides'], showWKT=True, showMeta=True)

    make_db(annotations)

