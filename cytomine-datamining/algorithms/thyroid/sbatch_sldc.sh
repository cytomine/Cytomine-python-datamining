#!/bin/bash
#SBATCH --job-name=sldc_test
#SBATCH --output=/home/mass/GRD/r.mormont/out/test/sldc_test.res
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=7G
#SBATCH --partition=Public
MODEL_PATH="/home/mass/GRD/r.mormont/models"
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-datamining/algorithms/thyroid/workflow.py \
    --cell_classifier "$MODEL_PATH/validated/incl_short.pkl" \
    --aggregate_classifier "$MODEL_PATH/validated/prolif_short.pkl" \
    --dispatch_classifier "$MODEL_PATH/validated/cpo.pkl" \
    --host "beta.cytomine.be" \
    --public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --software_id "152714969" \
    --project_id "186829908" \
    --slide_ids "186859011" \
    --tile_max_height "1024" \
    --tile_max_width "1024" \
    --working_path "/home/mass/GRD/r.mormont/nobackup/test_project/" \
    --base_path "/api/" \
    --verbose 10 \
    --n_jobs 7

# ",186858563,186851426,186851134,186850855,186850602,186850322,186849981,186849450,186848900,186848552,186847588,186847313,186845954,186845730,186845571,186845377,186845164,186844820,186844344,186843839,186843325,186842882,186842285,186842002,186841715,186841154" \
