# -*- coding: utf-8 -*-

import cytomine

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"

params = {
    'verbose' : False,
    'cytomine_host' : "beta.cytomine.be",
    'cytomine_public_key' : "ad014190-2fba-45de-a09f-8665f803ee0b",
    'cytomine_private_key' : "767512dd-e66f-4d3c-bb46-306fa413a5eb",
    'cytomine_base_path' : '/api/',
    'cytomine_working_path' : '/home/mass/GRD/r.mormont/tmp/check',
}


if __name__ == "__main__":
    ids = [18695733, 18695106, 18694972, 18694785, 18694034, 18693992, 18693950, 18693493, 18693435, 18693401, 18693386,
           18693363, 18255523, 18255403, 18255256, 18255197, 18255097, 18255022, 18254880, 18254850, 18254725, 18254694,
           18254678, 18254619, 18254503, 18254460, 18254434, 18254365, 18254234, 18254143, 18254122, 18254073, 18254042,
           18253943, 18253908, 18253725, 18253712, 18253695, 18253633, 18253604, 18253343, 18253318, 18253292, 18253107]

    conn = cytomine.Cytomine(params["cytomine_host"], params["cytomine_public_key"], params["cytomine_private_key"],
                             base_path=params['cytomine_base_path'], working_path=params['cytomine_working_path'],
                             verbose=params['verbose'])

    conn.get_image_instance(include_server_urls=True)
    for id in ids:
        annotation = conn.get_annotation(id)
        image = conn.get_image_instance(annotation.image)





