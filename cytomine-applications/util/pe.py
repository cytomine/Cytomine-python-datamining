# -*- coding: utf-8 -*-
from cytomine import Cytomine
from shapely.geometry import Polygon, Point, box
from shapely.ops import cascaded_union
from shapely.wkt import loads
import numpy as np

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


params = {
    'verbose' : False,
    'cytomine_host' : "beta.cytomine.be",
    'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cytomine_base_path' : '/api/',
    'cytomine_working_path' : '/home/mass/GRD/r.mormont/tmp/check',
    'project': 186829908,  # id of the project from which the annotations must be fetched
    'slides': # cells [ 186845954, 186845730, 186845571, 186845377, 186845164, 186844820, 186844344, 186843839, 186843325, 186842882, 186842285, 186842002, 186841715, 186841154, 186840535 ],
        [186859011, 186858563, 186851426, 186851134, 186850855, 186850602, 186850322, 186849981, 186849450, 186848900, 186848552, 186847588, 186847313],
        # all [186859011, 186858563, 186851426, 186851134, 186850855, 186850602, 186850322, 186849981, 186849450, 186848900, 186848552, 186847588, 186847313, 186845954, 186845730, 186845571, 186845377, 186845164, 186844820, 186844344, 186843839, 186843325, 186842882, 186842285, 186842002, 186841715, 186841154, 186840535],  # the slides that must be checked
    'terms': [676446,676390,676210,676434,675999,676026,676176,933004,676407],  # the ids of the user annotated terms to check
    'user': [179077547],  # ids of the users of which the annotation must be used as ground truth
    'job': 187595703  # ids of the jobs of which the annotation must be checked
}


def mk_circle(center, radius):
    return Point(center).buffer(radius)


def mk_empty_if(truth, estimated):
    if truth is None:
        truth = Polygon()
    if estimated is None:
        estimated = Polygon()
    return truth, estimated


def pe(truth, estimated, image_box):
    truth, estimated = mk_empty_if(truth, estimated)
    diff1 = estimated.difference(truth)
    diff2 = truth.difference(estimated)
    box_area = image_box.area
    box_area_squared = box_area * box_area
    pob = diff1.area * (box_area - truth.area) / box_area_squared
    pbo = diff2.area * truth.area / box_area_squared
    return 1 - (pbo + pob)


def recall(truth, estimated):
    truth, estimated = mk_empty_if(truth, estimated)
    return estimated.intersection(truth).area / truth.area


def balance(seg, image_box):
    return seg.area / image_box.area


def precision(truth, estimated):
    truth, estimated = mk_empty_if(truth, estimated)
    return estimated.intersection(truth).area / estimated.area


def fetch_geoms(cytomine, project, id_slides, id_users):
    annotations = cytomine.get_annotations(id_project=project, id_image=id_slides, id_user=id_users, showWKT=True, showMeta=True)
    geoms_dict = dict()
    for annotation in annotations:
        key = annotation.image
        polygons = geoms_dict.get(key, [])
        polygons.append(loads(annotation.location))
        geoms_dict[key] = polygons
    return geoms_dict

def set(conn, id, res, mag):
    inst = conn.get_image_instance(id)
    inst.resolution = res
    inst.magnification = mag
    conn.update(inst)

def fetch_user_job_id(cytomine, job_id):
    job = cytomine.get_job(job_id)
    user_job = cytomine.get_user(job.userJob)
    return user_job.id

if __name__ == "__main__":
    conn = Cytomine(params["cytomine_host"], params["cytomine_public_key"],
                    params["cytomine_private_key"], base_path = params['cytomine_base_path'],
                    working_path = params['cytomine_working_path'], verbose=True)

    all_truth = fetch_geoms(conn, params["project"], params["slides"], id_users=params["user"])
    all_estimated = fetch_geoms(conn, params["project"], params["slides"], id_users=fetch_user_job_id(conn, params["job"]))

    pes = []
    recalls = []
    precisions = []
    balances_e = []
    balances_t = []

    for id_slide in params["slides"]:
        truth, estimated = None, None
        if all_truth.has_key(id_slide):
            truth = cascaded_union(all_truth[id_slide])
        if all_estimated.has_key(id_slide):
            estimated = cascaded_union(all_estimated[id_slide])
        image = conn.get_image_instance(id_slide)
        image_box = box(0, 0, image.width, image.height)
        pe_ = pe(truth, estimated, image_box)
        recall_ = recall(truth, estimated)
        precision_ = precision(truth, estimated)
        balance_t_ = balance(truth, image_box)
        balance_e_ = balance(estimated, image_box)
        print "Image #{}".format(id_slide)
        print " - p_e       : {}".format(pe_)
        print " - recall    : {}".format(recall_)
        print " - precision : {}".format(precision_)
        print " - balance(t): {}".format(balance_t_)
        print " - balance(e): {}".format(balance_e_)
        print
        pes.append(pe_)
        recalls.append(recall_)
        precisions.append(precision_)
        balances_e.append(balance_e_)
        balances_t.append(balance_t_)

    print "Summary : "
    print " - average p_e       : {}".format(np.mean(np.array(pes)))
    print " - average recall    : {}".format(np.mean(np.array(recalls)))
    print " - average precision : {}".format(np.mean(np.array(precisions)))
    print " - average balance(t): {}".format(np.mean(np.array(balances_t)))
    print " - average balance(e): {}".format(np.mean(np.array(balances_e)))
