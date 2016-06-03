# -*- coding: utf-8 -*-
import optparse
import os

from util import CropTypeEnum, copy_content, create_dir, str2bool, str2list
from cytomine import cytomine
from pyxit import pyxitstandalone
from adapters import AnnotationCollectionAdapter
from mapper import BinaryMapper, DefaultMapper
from cytominejob import CytomineJob

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

class ModelBuilderJob(CytomineJob):

    def __init__(self, cytomine_client, software_id, project_id, output_mapper, dir_ls, pyxit_parameters, working_path,
                 zoom_level=1, reviewed_only=False, excluded_terms=None, users_only=None, excluded_images=None,
                 crop_type=CropTypeEnum.ALPHA_CROP):
        CytomineJob.__init__(self, cytomine_client, software_id, project_id)
        self._cytomine = cytomine_client
        self._working_path = working_path
        self._mapper = output_mapper
        self._reviewed_only = reviewed_only
        self._excluded_terms = excluded_terms
        self._users_only = users_only
        self._crop_type = crop_type
        self._zoom_level = zoom_level
        self._dir_ls = dir_ls
        self._pyxit_parameters = pyxit_parameters
        self._excluded_images = excluded_images

    def execute(self):
        annotations, dump_path = self._dump_annotations()
        self._generate_dir_ls_with_mapper(dump_path)
        self.set_progress(75, "Build model...")
        print "Build model..."
        pyxitstandalone.main(self._build_pyxit_argv())

    def _generate_dir_ls_with_mapper(self, src):
        create_dir(self._dir_ls, clean=True)
        outputs, terms = self._output_classes_from_dir_ls(src)
        # create output directories
        for output in outputs:
            create_dir(os.path.join(self._dir_ls, str(output)))
        # move crops from old directories to new ones
        for term in terms:
            old_dir = os.path.join(os.path.join(src, str(term)))
            new_dir = os.path.join(os.path.join(self._dir_ls, str(self._mapper.map(term))))
            copy_content(old_dir, new_dir)

    def _output_classes_from_dir_ls(self, src):
        terms = [int(o) for o in os.listdir(src) if os.path.isdir(os.path.join(src, o))]
        output_set = set()
        for term in terms:
            output_set.add(self._mapper.map(term))
        return list(output_set), terms

    def _dump_annotations(self):
        self.set_progress(25, "Get annotations...")
        print "Get annotations..."
        annotations = self._cytomine.get_annotations(id_project=self.project_id, reviewed_only=self._reviewed_only,
                                                     showMeta=True, id_user=self._users_only)
        excluded_terms = set(self._excluded_terms)
        excluded_images = set(self._excluded_images)
        filtered = [a for a in annotations.data()
                    if len(a.term) > 0
                    and set(a.term).isdisjoint(excluded_terms)
                    and a.image not in excluded_images]
        annotations = AnnotationCollectionAdapter(filtered)
        self.set_progress(25, "Dump annotations...")
        print "Dump annotations (total:{})...".format(annotations)
        dump_path = os.path.join(self._working_path, "alpha_crops")
        create_dir(dump_path)
        annotations = self._cytomine.dump_annotations(annotations=annotations, excluded_terms=self._excluded_terms,
                                                      get_image_url_func=CropTypeEnum.enum2crop(self._crop_type),
                                                      dest_path=dump_path, desired_zoom=self._zoom_level)
        return annotations, dump_path

    def _build_pyxit_argv(self):
        argv = []
        for key in self._pyxit_parameters:
            value = self._pyxit_parameters[key]
            if type(value) is bool or value == 'True':
                if bool(value):
                    argv.append("--%s" % key)
            elif not value == 'False':
                argv.append("--%s" % key)
                argv.append("%s" % value)
        return argv


def main(argv):
    p = optparse.OptionParser(description='Pyxit/Cytomine Classification model Builder',
                              prog='PyXit Classification Model Builder (PYthon piXiT)')

    p.add_option('--cytomine_host', type='string', default='', dest="cytomine_host",
                 help="The Cytomine host (eg: beta.cytomine.be, localhost:8080)")
    p.add_option('--cytomine_public_key', type='string', default='', dest="cytomine_public_key",
                 help="Cytomine public key")
    p.add_option('--cytomine_private_key', type='string', default='', dest="cytomine_private_key",
                 help="Cytomine private key")
    p.add_option('--cytomine_base_path', type='string', default='/api/', dest="cytomine_base_path",
                 help="Cytomine base path")
    p.add_option('--cytomine_working_path', default="/tmp/", type='string', dest="cytomine_working_path",
                 help="The working directory (eg: /tmp)")
    p.add_option('--cytomine_id_software', type="int", dest="cytomine_id_software",
                 help="The Cytomine software identifier")
    p.add_option('--cytomine_id_project', type="int", dest="cytomine_id_project",
                 help="The Cytomine project identifier")
    p.add_option('-z', '--cytomine_zoom_level', default=1, type='int', dest='cytomine_zoom_level',
                 help="working zoom level")
    p.add_option('--cytomine_excluded_terms', type='string', dest='cytomine_excluded_terms',
                 help="term ids of excluded terms")
    p.add_option('--cytomine_excluded_images', type='string', dest='cytomine_excluded_images',
                 help="image excluded from the learning set")
    p.add_option('--cytomine_selected_users', type='string', dest='cytomine_selected_users',
                 help="user from who the annotations should be taken")
    p.add_option('--cytomine_reviewed', type='string', default="False", dest="cytomine_reviewed",
                 help="Get reviewed annotations only")
    p.add_option('--cytomine_binary', type='string', default="False", dest="cytomine_binary",
                 help="True for binary classification")
    p.add_option('--cytomine_positive_terms', type='string', default='', dest="cytomine_positive_terms",
                 help="List of terms representing the positive class")
    p.add_option('--cytomine_negative_terms', type='string', default='', dest="cytomine_negative_terms",
                 help="List of terms representing the negative class")
    p.add_option('--cytomine_multiclass', type='string', default='False', dest="cytomine_multiclass",
                 help="True for multiclass classification")

    p.add_option('--pyxit_target_width', default=16, type='int', dest='pyxit_target_width',
                 help="pyxit subwindows width")
    p.add_option('--pyxit_target_height', default=16, type='int', dest='pyxit_target_height',
                 help="pyxit subwindows height")
    p.add_option('--pyxit_colorspace', default=2, type='int', dest='pyxit_colorspace',
                 help="pyxit colorspace encoding")
    p.add_option('--pyxit_n_jobs', type='int', dest='pyxit_n_jobs',
                 help="pyxit number of jobs for trees")
    p.add_option('--pyxit_n_subwindows', default=10, type="int", dest="pyxit_n_subwindows",
                 help="number of subwindows")
    p.add_option('--pyxit_min_size', default=0.1, type="float", dest="pyxit_min_size",
                 help="min size")
    p.add_option('--pyxit_max_size', default=0.9, type="float", dest="pyxit_max_size",
                 help="max size")
    p.add_option('--pyxit_interpolation', default=2, type="int", dest="pyxit_interpolation",
                 help="interpolation method 1,2,3,4")
    p.add_option('--pyxit_transpose', type='string', default="False", dest="pyxit_transpose",
                 help="transpose subwindows")
    p.add_option('--pyxit_fixed_size', type='string', default="False", dest="pyxit_fixed_size",
                 help="extract fixed size subwindows")
    p.add_option('--pyxit_save_to', type='string', default="/tmp", dest="pyxit_save_to",
                 help="pyxit_save_to")

    p.add_option('--forest_n_estimators', default=10, type="int", dest="forest_n_estimators",
                 help="number of base estimators (T)")
    p.add_option('--forest_max_features', default=1, type="int", dest="forest_max_features",
                 help="max features at test node (k)")
    p.add_option('--forest_min_samples_split', default=1, type="int", dest="forest_min_samples_split",
                 help="minimum node sample size (nmin)")
    p.add_option('--svm', default=0, dest="svm",
                 help="final svm classifier: 0=nosvm, 1=libsvm, 2=liblinear, 3=lr-l1, 4=lr-l2", type="int")
    p.add_option('--svm_c', default=1.0, type="float", dest="svm_c",
                 help="svm C")
    p.add_option('--verbose', type='string', default="False", dest="verbose",
                 help="Turn on (1) or off (0) verbose mode")

    options, arguments = p.parse_args(args=argv)
    print options
    print arguments

    parameters = {
        'cytomine_host': options.cytomine_host,  # "beta.cytomine.be",
        'cytomine_public_key': options.cytomine_public_key,  # "ad014190-2fba-45de-a09f-8665f803ee0b",
        'cytomine_private_key': options.cytomine_private_key,  # "767512dd-e66f-4d3c-bb46-306fa413a5eb",
        'cytomine_base_path': options.cytomine_base_path,  # '/api/',
        'cytomine_working_path': options.cytomine_working_path,  # '/home/mass/GRD/r.mormont/nobackup/validation',
        'cytomine_verbose': str2bool(options.verbose),  # True
        'cytomine_zoom_level': options.cytomine_zoom_level,  # 1,
        'cytomine_reviewed': str2bool(options.cytomine_reviewed),  # False,
        'cytomine_id_project': options.cytomine_id_project,  # 716498,
        'cytomine_id_software': options.cytomine_id_software,  # 179703916,
        'cytomine_users': str2list(options.cytomine_selected_users),  # [671279],  # C. Degand # me : [179077547],  #
        'cytomine_excluded_terms': str2list(options.cytomine_excluded_terms),  # see below
        'cytomine_positive': str2list(options.cytomine_positive_terms),
        'cytomine_negative': str2list(options.cytomine_negative_terms),
        'cytomine_excluded_images': str2list(options.cytomine_excluded_images)
    }
    # [676446,676390,676210,676434,676176,676407,15109451,15109483,15109489,15109495],  # cells
    # [675999, 676026, 933004],  # patterns
    pyxit_parameters = {
        'pyxit_target_width': options.pyxit_target_width,  # 16,
        'pyxit_target_height': options.pyxit_target_height,  # 16,
        'pyxit_n_subwindows': options.pyxit_n_subwindows,  # 20,
        'pyxit_min_size': options.pyxit_min_size,  # 0.1,
        'pyxit_max_size': options.pyxit_max_size,  # 0.9,
        'pyxit_colorspace': options.pyxit_colorspace,  # 2,
        'pyxit_interpolation': options.pyxit_interpolation,  # 1,
        'pyxit_transpose': str2bool(options.pyxit_transpose),  # None,
        'pyxit_fixed_size': str2bool(options.pyxit_fixed_size),  # None,
        'forest_n_estimators': options.forest_n_estimators,  # 50,
        'forest_max_features': options.forest_max_features,  # 28,
        'forest_min_samples_split': options.forest_min_samples_split,  # 10,
        'svm': options.svm,  # False
        'svm_c': options.svm_c,
        'pyxit_n_jobs': options.pyxit_n_jobs,  # 10
        'pyxit_save_to': options.pyxit_save_to,
        'dir_ls': os.path.join(options.cytomine_working_path, "ls")
    }

    client = cytomine.Cytomine(parameters["cytomine_host"],
                               parameters["cytomine_public_key"],
                               parameters["cytomine_private_key"],
                               base_path=parameters['cytomine_base_path'],
                               working_path=parameters['cytomine_working_path'],
                               verbose=parameters['cytomine_verbose'])

    # define class mapper
    if str2bool(options.cytomine_binary):
        mapper = BinaryMapper(parameters["cytomine_positive"], parameters["cytomine_negative"])
    elif str2bool(options.cytomine_multiclass):
        mapper = DefaultMapper()
    else:
        mapper = DefaultMapper()

    print "Pyxit parameters : {}".format(pyxit_parameters)
    print "Parameters : {}".format(parameters)
    with ModelBuilderJob(client, parameters["cytomine_id_software"], parameters["cytomine_id_project"],
                         mapper, pyxit_parameters["dir_ls"], pyxit_parameters, parameters["cytomine_working_path"],
                         zoom_level=parameters["cytomine_zoom_level"], reviewed_only=parameters["cytomine_reviewed"],
                         excluded_terms=parameters["cytomine_excluded_terms"],
                         users_only=parameters["cytomine_users"],
                         excluded_images=parameters["cytomine_excluded_images"]) as job:
        job.execute()


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
