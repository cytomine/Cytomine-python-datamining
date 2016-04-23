#!/bin/bash
#SBATCH --job-name=build_patterns_prolif_vs_norm
#SBATCH --output=/home/mass/GRD/r.mormont/out/build/build_patterns_prolif_vs_norm.res
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=Public
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/custom_validation/build.py \
    --cytomine_host "beta.cytomine.be" \
    --cytomine_public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --cytomine_private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --cytomine_base_path "/api/" \
    --cytomine_working_path "/home/mass/GRD/r.mormont/nobackup/build/patterns_prolif_vs_norm" \
    --cytomine_id_software 30397100 \
    --cytomine_id_project 716498 \
    --cytomine_excluded_terms "676446,676390,676210,676434,676176,676407,8844862,8844845,9444456,15054705,15054765,15109451,15109483,15109489,15109495,22042230,28792193,30559888" \
    --cytomine_selected_users "671279" \
    --cytomine_binary "True" \
    --cytomine_positive_terms "676026,933004" \
    --cytomine_negative_terms "675999" \
    --pyxit_n_jobs 10 \
    --pyxit_n_subwindows 50 \
    --pyxit_min_size 0.1 \
    --pyxit_max_size 0.75 \
    --pyxit_save_to "/home/mass/GRD/r.mormont/models/patterns_prolif_vs_norm.pkl" \
    --forest_n_estimators 100 \
    --forest_max_features 16 \
    --verbose "True"
