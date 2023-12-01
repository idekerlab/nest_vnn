#!/bin/bash
#SBATCH --job-name=NeST_VNN
#SBATCH --output=out.log
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --dependency=singleton

bash "/cellar/users/asinghal/Workspace/nest_vnn/sample/train.sh"
