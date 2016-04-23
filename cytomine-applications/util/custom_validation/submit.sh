#!/bin/bash
SH_PATH="$HOME/sftp/cytomine-applications/util/custom_validation"
sbatch -p Public "$SH_PATH/sbatch_validation_cells_vs_patterns.sh"
sbatch -p Public "$SH_PATH/sbatch_validation_cells_vs_all.sh"
sbatch -p Public "$SH_PATH/sbatch_validation_patterns_vs_all.sh"
