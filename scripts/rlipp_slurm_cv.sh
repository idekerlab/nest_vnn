#!/bin/bash
#SBATCH --partition=nrnb-compute
#SBATCH --account=nrnb
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --dependency=singleton

cpu_count=8

bash "${1}/scripts/rlipp_cv.sh" $1 $2 $3 $4 $5 $6 $7 $cpu_count
