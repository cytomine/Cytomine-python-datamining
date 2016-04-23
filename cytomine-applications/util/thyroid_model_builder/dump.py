import cytomine
from cytomine.models import Annotation

parameters = {
'cytomine_host' : "beta.cytomine.be",
'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
'cytomine_base_path' : "/api/",
'cytomine_working_path' : '/home/vagrant/my_dump/',
'cytomine_id_project' : 716498,
'cytomine_id_image': 8120370,
'cytomine_zoom_level' : 1,
'cytomine_dump_type' : 1,
'cytomine_annotation_projects' : ["716498"],  #id of projets from which we dump annotations for learning
'cytomine_excluded_terms' : ["676446", "676390", "676210", "676434", "675999", "676026", "676176", "933004",
                             "676407", "8844862", "8844845", "9444456", "15054705", "15054765", "15109451",
                             "15109483", "15109489", "15109495", "28792193", "30559888"],
                            #exclude these term ids
}


conn = cytomine.Cytomine(parameters["cytomine_host"],
                         parameters["cytomine_public_key"],
                         parameters["cytomine_private_key"] ,
                         base_path = parameters['cytomine_base_path'],
                         working_path = parameters['cytomine_working_path'],
                         verbose=True)

annotations = conn.get_annotations(id_project=parameters["cytomine_id_project"],
                                   id_image=parameters["cytomine_id_image"],
                                   id_term="22042230")
annotation_get_func = Annotation.get_annotation_crop_url
annotations=conn.dump_annotations(annotations = annotations, get_image_url_func = annotation_get_func,
                                  dest_path = parameters["cytomine_working_path"],
                                  desired_zoom = parameters['cytomine_zoom_level'],
                                  excluded_terms=parameters['cytomine_excluded_terms'])
