#!/bin/bash
#SBATCH --job-name=validation_patterns_vs_all
#SBATCH --output=/home/mass/GRD/r.mormont/out/validation/validation_patterns_vs_all.res
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --partition=Public
MY_HOME="/home/mass/GRD/r.mormont"
MODEL_PATH="$MY_HOME/models"
/home/mass/GRD/r.mormont/miniconda/bin/python /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/custom_validation/validation.py \
    "$MODEL_PATH/
     "beta.cytomine.be" \
    --public_key "ad014190-2fba-45de-a09f-8665f803ee0b" \
    --private_key "767512dd-e66f-4d3c-bb46-306fa413a5eb" \
    --base_path "/api/" \
    --working_path "/home/mass/GRD/r.mormont/nobackup/validation/patterns_vs_all" \
    --id_software 179703916 \
    --id_project 716498 \
    --excluded_terms "9444456,22042230,28792193,30559888,15054705,15054765" \
    --selected_users "671279" \
    --binary "True" \
    --positive_terms "675999,676026,933004" \
    --negative_terms "676446,676390,676210,676434,676176,676407,15109451,15109483,15109489,15109495,8844862,8844845" \
    --pyxit_n_jobs 10 \
    --pyxit_n_subwindows 50 \
    --pyxit_min_size 0.1 \
    --pyxit_max_size 0.6 \
    --forest_n_estimators 100 \
    --forest_max_features 16 \
    --cv_k_folds 10 \
    --verbose "True"



    parser = argparse.ArgumentParser()  # TODO desc.
    parser.add_argument("cell_classifier",           help="File where the cell classifier has been pickled")
    parser.add_argument("aggregate_classifier",      help="File where the architectural pattern classifier has been pickled")
    parser.add_argument("cell_dispatch_classifier",  help="File where the cell dispatch classifier has been pickled")
    parser.add_argument("aggregate_dispatch_classifier", help="File where the aggregate dispatch classifier has been pickled")
    parser.add_argument("host",                      help="Cytomine server host URL")
    parser.add_argument("public_key",                help="User public key")
    parser.add_argument("private_key",               help="User Private key")
    parser.add_argument("software_id",               help="Identifier of the software on the Cytomine server")
    parser.add_argument("project_id",                help="Identifier of the project to process on the Cytomine server")
    parser.add_argument("slide_ids",                 help="Sequence of ids of the slides to process", nargs="+", type=int)
    parser.add_argument("--working_path",            help="Directory for caching temporary files", default="/tmp")
    parser.add_argument("--protocol",                help="Communication protocol",default="http://")
    parser.add_argument("--base_path",               help="n/a", default="/api/")
    parser.add_argument("--timeout",                 help="Timeout time for connection (in seconds)", type=positive_int, default="120")
    parser.add_argument("--verbose",                 help="increase output verbosity", action="store_true", default=True)
    parser.add_argument("--nb_jobs",                 help="Number of core to use", type=not_zero, default=1)