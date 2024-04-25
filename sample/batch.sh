#!/bin/bash
#SBATCH --job-name=NeST_VNN
#SBATCH --output=out.log
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --dependency=singleton

homedir="/cellar/users/asinghal/Workspace/nest_vnn"

#bash "${homedir}/sample/train.sh" "${homedir}"

bash "${homedir}/sample/test_palbociclib.sh" "${homedir}"
