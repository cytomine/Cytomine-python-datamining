#!/bin/bash
#SBATCH --job-name=validation_cells_vs_all
#SBATCH --output=/home/mass/GRD/r.mormont/out/validation/validation_cells_vs_all.res
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=Public
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/custom_validation/validation.py \
    --cytomine_host "beta.cytomine.be" \
    --cytomine_public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --cytomine_private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --cytomine_base_path "/api/" \
    --cytomine_working_path "/home/mass/GRD/r.mormont/nobackup/validation/cells_vs_all" \
    --cytomine_id_software 179703916 \
    --cytomine_id_project 716498 \
    --cytomine_excluded_terms "9444456,22042230,28792193,30559888,15054705,15054765" \
    --cytomine_selected_users "671279" \
    --cytomine_binary "True" \
    --cytomine_positive_terms "676446,676390,676210,676434,676176,676407,15109451,15109483,15109489,15109495" \
    --cytomine_negative_terms "675999,676026,933004,8844862,8844845" \
    --pyxit_n_jobs 10 \
    --pyxit_n_subwindows 50 \
    --pyxit_min_size 0.6 \
    --pyxit_max_size 1.0 \
    --forest_n_estimators 100 \
    --forest_max_features 16 \
    --cv_k_folds 10 \
    --verbose "True"



