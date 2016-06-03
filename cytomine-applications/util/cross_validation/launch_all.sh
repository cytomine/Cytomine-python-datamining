#!/bin/bash
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/incl_vs_norm_et.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/incl_vs_norm_svm.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/incl_vs_norm_reviewed_et.sh
sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/incl_vs_norm_reviewed_svm.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/prolif_vs_norm_et.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/prolif_vs_norm_svm.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/prolif_vs_norm_reviewed_et.sh
sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/prolif_vs_norm_reviewed_svm.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/cpo_et.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/cpo_svm.sh
sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/cpo_reviewed_svm.sh
##sbatch -p Cytomine /home/mass/GRD/r.mormont/sftp/cytomine-applications/util/cross_validation/cpo_reviewed_et.sh
