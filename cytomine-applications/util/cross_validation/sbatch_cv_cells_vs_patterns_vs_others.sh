#!/bin/bash
#SBATCH --job-name=cv_cpo3
#SBATCH --output=/home/mass/GRD/r.mormont/out/validation/cv_cpo3.res
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --time=192:00:00
#SBATCH --mem=700G
#SBATCH --partition=Cytomine
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/pyxit_cross_validator.py \
    --cytomine_host "beta.cytomine.be" \
    --cytomine_public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --cytomine_private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --cytomine_base_path "/api/" \
    --cytomine_working_path "/home/mass/GRD/r.mormont/nobackup/cv" \
    --cytomine_id_software 179703916 \
    --cytomine_id_project 716498 \
    --cytomine_selected_users 671279 \
    --cytomine_binary "True" \
    --cytomine_excluded_terms 9444456 \
        --cytomine_excluded_terms 22042230 \
        --cytomine_excluded_terms 28792193 \
        --cytomine_excluded_terms 30559888 \
        --cytomine_excluded_terms 15054705 \
        --cytomine_excluded_terms 15054765 \
    --cytomine_positive_terms 676446 \
        --cytomine_positive_terms 676390 \
        --cytomine_positive_terms 676210 \
        --cytomine_positive_terms 676434 \
        --cytomine_positive_terms 676176 \
        --cytomine_positive_terms 676407 \
        --cytomine_positive_terms 15109483 \
    --cytomine_negative_terms 675999 \
        --cytomine_negative_terms 676026 \
        --cytomine_negative_terms 933004 \
    --cytomine_other_terms 8844862 \
        --cytomine_other_terms 8844845 \
        --cytomine_other_terms 15109451 \
        --cytomine_other_terms 15109489 \
        --cytomine_other_terms 15109495 \
    --cytomine_excluded_annotations 30675573 \
        --cytomine_excluded_annotations 18107252 \
        --cytomine_excluded_annotations 9321884 \
        --cytomine_excluded_annotations 7994253 \
        --cytomine_excluded_annotations 9313842 \
    --cytomine_test_images 8124112 \
        --cytomine_test_images 8123867 \
        --cytomine_test_images 8122868 \
        --cytomine_test_images 8122830 \
        --cytomine_test_images 8120497 \
        --cytomine_test_images 8120408 \
        --cytomine_test_images 8120321 \
        --cytomine_test_images 728799 \
        --cytomine_test_images 728744 \
        --cytomine_test_images 728725 \
        --cytomine_test_images 728709 \
        --cytomine_test_images 728689 \
        --cytomine_test_images 728675 \
        --cytomine_test_images 728391 \
        --cytomine_test_images 724858 \
        --cytomine_test_images 719625 \
        --cytomine_test_images 716534 \
        --cytomine_test_images 716528 \
    --cytomine_verbose 0 \
    --cv_images_out 1 \
    --pyxit_n_jobs 40 \
    --pyxit_dir_ls "/home/mass/GRD/r.mormont/nobackup/cv/ls" \
    --forest_n_estimators 20 \
    --pyxit_n_subwindows 100 \
    --pyxit_colorspace 1 \
        --pyxit_colorspace 2 \
    --pyxit_min_size 0.1 \
        --pyxit_min_size 0.3 \
        --pyxit_min_size 0.5 \
    --pyxit_max_size 1.0 \
        --pyxit_max_size 0.8 \
        --pyxit_max_size 0.6 \
    --forest_min_samples_split 578 \
    --forest_max_features 1 \
        --forest_max_features 28 \
        --forest_max_features 384 \
        --forest_max_features 768 \
    --svm 0 \
    --svm_c 0.1
