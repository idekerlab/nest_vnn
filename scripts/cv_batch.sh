#!/bin/bash
#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --dependency=singleton

bash "${1}/scripts/cv_train.sh" $1 $2 $3 $4 $5 $6 $7
bash "${1}/scripts/cv_test.sh" $1 $2 $3 $4 $5 $6 $7
if [ $4 = "Palbociclib" ]
then
    bash "${1}/scripts/cv_test_genie.sh" $1 $2 $3 $4 $5 $6
fi

#bash "${1}/scripts/cv_test_shuffled_input.sh" $1 $2 $3 $4 $5 $6 $7
