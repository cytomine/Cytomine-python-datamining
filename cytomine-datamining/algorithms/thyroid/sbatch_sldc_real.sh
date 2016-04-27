#!/bin/bash
#SBATCH --job-name=sldc_real_slide_8122868
#SBATCH --output=/home/mass/GRD/r.mormont/out/sldc/sldc_real_slide_8122868.res
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=192:00:00
#SBATCH --mem=100G
#SBATCH --partition=Public
MODEL_PATH="/home/mass/GRD/r.mormont/models"
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-datamining/algorithms/thyroid/workflow.py \
    --cell_classifier "$MODEL_PATH/patterns_prolif_vs_norm.pkl" \
    --aggregate_classifier "$MODEL_PATH/cells_inclusion_vs_norm.pkl" \
    --cell_dispatch_classifier "$MODEL_PATH/cells_reduced_vs_all.pkl"  \
    --aggregate_dispatch_classifier "$MODEL_PATH/patterns_vs_all.pkl" \
    --host "beta.cytomine.be" \
    --public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --software_id "152714969" \
    --project_id "716498" \
    --slide_ids "8122868" \
    --tile_max_height "2048" \
    --tile_max_width "2048" \
    --working_path "/home/mass/GRD/r.mormont/nobackup/sldc/" \
    --base_path "/api/" \
    --n_jobs 10
