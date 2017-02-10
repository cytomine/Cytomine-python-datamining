#!/bin/bash

python add_and_run_job.py \
    --cytomine_host "demo.cytomine.be" \
    --cytomine_public_key "XXX" \
    --cytomine_private_key "XXX" \
    --cytomine_base_path "/api/" \
    --cytomine_working_path  "/tmp/cytomine" \
    --cytomine_id_software 19718236 \
    --cytomine_id_project 526946 \
    --cytomine_id_image 527252 \
    --sldc_tile_overlap 7 \
    --sldc_tile_width 768 \
    --sldc_tile_height 768 \
    --pyxit_model_path "model.pkl" \
    --n_jobs 4 \
    --min_area 500 \
    --threshold 215 \
    --rseed 0 \
    --working_path "/tmp/sldc"
