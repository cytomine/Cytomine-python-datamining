
import cytomine
from shapely import wkt
import numpy as np

params = {
    'verbose' : False,
    'cytomine_host' : "beta.cytomine.be",
    'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cytomine_base_path' : '/api/',
    'cytomine_working_path' : '/home/vagrant/tmp',
    'project': 151860018,  # id of the project from which the annotations must be fetched
    'slides': [151870700,151870615,151870539,151870465,151870433,151870321,151870170,151870070,151869994,151869936],  # the slides that must be checked
    'terms': [676026],  # the ids of the user annotated terms to check
    'users': [179077547],  # ids of the users of which the annotation must be used as ground truth
    'jobs': [186525452]  # ids of the jobs of which the annotation must be checked
}

thyroid_ontology = {
	676446: "Cell NOS",
	676390: "Cell incl.",
	676210: "Cell artifact",
	676434: "Cell ground glass",
	675999: "Pattern norm.",
	676026: "Pattern prolif.",
	676176: "Cell norm.",
	933004: "Pattern prolif. (minor)",
	676407: "Cell grooves"
}


def create_or_add(_dict, key, elem):
    if key in _dict:
        _dict[key] += [elem]
    else:
        _dict[key] = [elem]


def aggregate_by_image(annotations, collection=True):
    aggregation = {}
    for annotation in annotations.data() if collection else annotations:
        create_or_add(aggregation, annotation.image, annotation)
    return aggregation


def aggregate_by_term(annotations, collection=True):
    """
    Group annotations having the same associated term together.

    :param annotations:
    :return:
    """
    aggregation = {}
    for annotation in annotations.data() if collection else annotations:
        for term in annotation.term:
            create_or_add(aggregation, term, annotation)
    return aggregation


class ConfusionMatrix(object):

    def __init__(self, ids, names_map=None):
        self._ids = ids
        self._matrix = np.zeros((len(ids), len(ids)))
        self._ids_indexes_lkup = {}
        self._names_map = names_map
        cnt = 0
        for id in ids:
            self._ids_indexes_lkup[id] = cnt
            cnt += 1

    def add(self, true_value, pred_value, count=1):
        i, j = self._ids_indexes_lkup[true_value], self._ids_indexes_lkup[pred_value]
        self._matrix[i,j] += count

    def __str__(self):
        headers = list(map(str, self._ids)) if self._names_map is None else list(map(lambda id: self._names_map[id], self._ids))
        repre = ';' + ';'.join(headers) + '\n'
        for i, header in enumerate(headers):
            repre += header + ";"
            repre += ';'.join(list(map(str, self._matrix[i,:].tolist())))
            repre += "\n"
        return repre

    def add_mat(self, confusion_matrix):
        if len(self._ids) != len(confusion_matrix._ids):
            raise RuntimeError("dimensions mismatch between confusion matrices")
        self._matrix += confusion_matrix._matrix

class IntersectionCache(object):
    """
    Keep track of already computed polygon intersection not to compute them again
    """
    def __init__(self):
        self._cache = {}

    def _cache_has(self, id1, id2):
        key1, key2 = IntersectionCache._to_keys(id1, id2)
        return key1 in self._cache or key2 in self._cache

    def _cache_get(self, id1, id2):
        key1, key2 = IntersectionCache._to_keys(id1, id2)
        if key1 in self._cache:
            return self._cache[key1]
        elif key2 in self._cache:
            return self._cache[key2]
        else:
            return None

    def _cache_set(self, id1, id2, value):
        key = IntersectionCache._to_key(id1, id2)
        self._cache[key] = value

    def intersection(self, annot1, annot2):
        if self._cache_has(annot1.id, annot2.id):
            return self._cache_get(annot1.id, annot2.id)

        ## not found in cache so compute the intersection
        polygon1 = wkt.loads(annot1.location)
        polygon2 = wkt.loads(annot2.location)

        if polygon1.intersects(polygon2):
            intersection = polygon1.intersection(polygon2)
        else:
            intersection = False

        self._cache_set(annot1.id, annot2.id, intersection)
        return intersection

    @classmethod
    def _to_key(cls, id1, id2):
        return "{},{}".format(id1, id2)

    @classmethod
    def _to_keys(cls, id1, id2):
        return cls._to_key(id1, id2), cls._to_key(id2, id1)


class AnnotReport(object):

    def __init__(self, id_user_annot, id_algo_annot=None, cover=None):
        self._id_user_annot = id_user_annot
        self._id_algo_annot = id_algo_annot
        self._cover = cover

    def do_match(self):
        return self._id_algo_annot is not None

    @property
    def cover(self):
        return self._cover

    @classmethod
    def format_cover(cls, cover):
        return list(map(lambda cover_item: "{}%".format(round(cover_item * 100, 2)), cover))

    def __str__(self):
        if self._id_algo_annot is not None:
            return "User annot. #{}, algo annot. #{} : {}".format(self._id_user_annot, self._id_algo_annot, AnnotReport.format_cover(self._cover))
        else:
            return "User annot. #{} : no matching algo annotation".format(self._id_user_annot)


class SlideReport(object):

    def __init__(self, slide_id, user_annotations, algo_annotations, intersection_cache):
        self._slide_id = slide_id
        self._user_annots_by_term = aggregate_by_term(user_annotations, collection=False)
        self._user_annots = user_annotations
        self._algo_annots_by_term = aggregate_by_term(algo_annotations, collection=False)
        self._algo_annots = algo_annotations
        self._intersection_cache = intersection_cache
        self._loc_test = {}
        self._compute()

    def _localization_test(self):
        """
        Test annotations without looking if the terms match (only look at terms of user annotations)
        """
        for term in self._user_annots_by_term.keys(): # iter over user annotated terms
            term_test_list = []
            for user_annot in self._user_annots_by_term[term]: # iter user annotation for a given term
                curr_user_annot_lst = []
                user_polygon = wkt.loads(user_annot.location)

                if term in self._algo_annots_by_term:
                    for algo_annot in self._algo_annots_by_term[term]: # iter over algo annotation of the slide
                        algo_polygon = wkt.loads(algo_annot.location)
                        inter_poly = self._intersection_cache.intersection(user_annot, algo_annot)
                        if inter_poly is not False:
                            user_area = user_polygon.area
                            algo_area = algo_polygon.area
                            inte_area = inter_poly.area
                            user_ratio = inte_area / user_area
                            algo_ratio = inte_area / algo_area
                            curr_user_annot_lst.append(AnnotReport(user_annot.id, id_algo_annot=algo_annot.id,
                                                                   cover=[user_ratio, 1 - user_ratio,
                                                                          algo_ratio, 1 - algo_ratio]))
                if len(curr_user_annot_lst) == 0:
                    curr_user_annot_lst.append(AnnotReport(user_annot.id))
                term_test_list += curr_user_annot_lst
            self._loc_test[term] = term_test_list

    def confusion(self):
        MISSED = 0  # id of the algo annotations that doesn't match any user annotations
        terms = thyroid_ontology.keys() + [MISSED]
        names = thyroid_ontology.copy()
        names[0] = "Missed"
        confusion_matrix = ConfusionMatrix(terms, names_map=names)

        for algo_annot in self._algo_annots:
            user_annot = self._find_user_annot(algo_annot)
            if user_annot is None: # no user annotation
                confusion_matrix.add(MISSED, algo_annot.term[0])
            else:
                confusion_matrix.add(user_annot.term[0], algo_annot.term[0])

        return confusion_matrix

    def _find_user_annot(self, algo_annot):
        # find a user annot that have the same term and that intersects the user annot
        # return None if none was found
        term = algo_annot.term[0] # get first term
        if term not in self._user_annots_by_term:
            return None
        user_annotations = self._user_annots_by_term[term]
        for user_annot in user_annotations:
            intersec = self._intersection_cache.intersection(algo_annot, user_annot)
            if intersec is not False:
                return user_annot
        return None

    def _compute(self):
        self._localization_test()

    def slide_summary(self):
        not_matching_count = 0
        cover_summary = []

        for term in self._loc_test:
            for test in self._loc_test[term]:
                if test.do_match():
                    cover_summary.append(test.cover)
                else:
                    not_matching_count += 1

        cover_summary = np.mean(cover_summary, axis=0)
        return not_matching_count, AnnotReport.format_cover(cover_summary)

    def __str__(self):
        descr =  " ***  Slide #{}  ***\n".format(self._slide_id)
        descr += "Summary : \n"
        not_matched_count, cover_summary = self.slide_summary()
        descr += " - Not matched : {}\n".format(not_matched_count)
        descr += " - Matched summary : {}\n".format(cover_summary)
        descr += " Annotations per term : \n"
        for term in self._loc_test:
            descr += " - Term '{}' (#{})\n".format(thyroid_ontology[term], term)
            for test in self._loc_test[term]:
                descr += "   * {}\n".format(str(test))

        return descr


class JobReport(object):

    def __init__(self, user_annotations, algo_annotations, intersection_cache):
        self._user_annots = user_annotations
        self._user_annots_by_slide = aggregate_by_image(user_annotations)
        self._algo_annots = algo_annotations
        self._algo_annots_by_slide = aggregate_by_image(algo_annotations)
        self._intersection_cache = intersection_cache


    def __str__(self):
        repr = ""
        confusion_matrix = None
        for slide in self._user_annots_by_slide:
            slide_report = SlideReport(slide, self._user_annots_by_slide[slide],
                                       self._algo_annots_by_slide[slide], self._intersection_cache)
            repr += "{}\n".format(str(slide_report))
            if confusion_matrix is None:
                confusion_matrix = slide_report.confusion()
            else:
                confusion_matrix.add_mat(slide_report.confusion())
        repr += "\n{}\n".format(str(confusion_matrix))
        return repr

if __name__ == "__main__":
    conn = cytomine.Cytomine(params["cytomine_host"], params["cytomine_public_key"],
                             params["cytomine_private_key"], base_path = params['cytomine_base_path'],
                             working_path = params['cytomine_working_path'], verbose=params['verbose'])

    intersection_cache = IntersectionCache()

    # download jobs annotation
    truth_annotations = conn.get_annotations(id_project=params['project'], id_user=params['users'],
                                             id_image=params['slides'], showWKT=True, showMeta=True)


    # for each job lauchned, evaluate the performances
    for job_id in params['jobs']:
        job = conn.get_job(job_id)
        user_job = conn.get_user(job.userJob)
        test_annotations = conn.get_annotations(id_project=params['project'], id_image=params['slides'],
                                                id_user=user_job.id, showWKT=True, showMeta=True)
        print(str(JobReport(truth_annotations, test_annotations, intersection_cache)))
