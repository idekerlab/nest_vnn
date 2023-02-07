#!/bin/bash
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --dependency=singleton

#bash "${1}/scripts/train.sh" $1 $2 $3 $4 $5
#bash "${1}/scripts/test.sh" $1 $2 $3 $4 $5
bash "${1}/scripts/test_pdx.sh" $1 $2 $3 $4 $5
if [ $4 = "Palbociclib" ]
then
	bash "${1}/scripts/test_genie.sh" $1 $2 $3 $4 $5
fi
