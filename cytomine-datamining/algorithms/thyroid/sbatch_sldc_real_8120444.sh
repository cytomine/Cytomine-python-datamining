#!/bin/bash
#SBATCH --job-name=sldc_real_slides_8120444_3
#SBATCH --output=/home/mass/GRD/r.mormont/out/sldc/sldc_real_slides_8120444_3.res
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --time=192:00:00
#SBATCH --mem=2000G
#SBATCH --partition=Cytomine
MODEL_PATH="/home/mass/GRD/r.mormont/models"
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-datamining/algorithms/thyroid/workflow.py \
    --cell_classifier "$MODEL_PATH/validated/final/incl_vs_norm_reviewed_et.pkl" \
    --aggregate_classifier "$MODEL_PATH/validated/final/prolif_vs_norm_reviewed_et.pkl" \
    --dispatch_classifier "$MODEL_PATH/validated/final/cpo_reviewed_svm.pkl" \
    --host "beta.cytomine.be" \
    --public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --software_id "152714969" \
    --project_id "716498" \
    --slide_ids "8120444" \
    --tile_max_height "1024" \
    --tile_max_width "1024" \
    --working_path "/home/mass/GRD/r.mormont/nobackup/sldc/" \
    --base_path "/api/" \
    --n_jobs 100
