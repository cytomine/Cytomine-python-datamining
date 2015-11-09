__author__ = 'Romain'

import cytomine

parameters = {
    'cytomine_host' : "beta.cytomine.be",
    'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cytomine_base_path' : '/api/',
    'cytomine_working_path' : '/home/vagrant/downloaded/annotations',
    'cytomine_id_jobs' : [183905946,183783181]
}

conn = cytomine.Cytomine(parameters["cytomine_host"],
                         parameters["cytomine_public_key"],
                         parameters["cytomine_private_key"] ,
                         base_path = parameters['cytomine_base_path'],
                         working_path = parameters['cytomine_working_path'],
                         verbose=True)

for job_id in parameters["cytomine_id_jobs"]:
    job = conn.get_job(job_id)
    conn.update_job_status(job, status = 4, status_comment = "Script error...", progress = 100)
