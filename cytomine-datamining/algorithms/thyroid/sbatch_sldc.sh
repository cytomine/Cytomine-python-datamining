#!/bin/bash
#SBATCH --job-name=sldc_test2
#SBATCH --output=/home/mass/GRD/r.mormont/out/validation/sldc_test2.res
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=Public
HM=/home/mass/GRD/r.mormont
MODEL_PATH="$HM/models"
$HM/miniconda/bin/python $HM/sftp/cytomine-datamining/algorithms/thyroid/workflow.py \
    --cell_classifier "$MODEL_PATH/patterns_prolif_vs_norm.pkl" \
    --aggregate_classifier "$MODEL_PATH/cells_inclusion_vs_norm.pkl" \
    --cell_dispatch_classifier "$MODEL_PATH/cells_reduced_vs_all.pkl"  \
    --aggregate_dispatch_classifier "$MODEL_PATH/patterns_vs_all.pkl" \
    --host "beta.cytomine.be" \
    --public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --software_id "152714969" \
    --project_id "186829908" \
    --slide_ids "186859011,186858563,186851426,186851134,186850855,186850602,186850322,186849981,186849450,186848900,186848552,186847588,186847313,186845954,186845730,186845571,186845377,186845164,186844820,186844344,186843839,186843325,186842882,186842285,186842002,186841715,186841154" \
    --tile_max_height "2048" \
    --tile_max_width "2048" \
    --working_path "$HM/tmp/sldc/" \
    --base_path "/api/" \
    --verbose 0 \
    --n_jobs 10
