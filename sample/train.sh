#!/bin/bash

#SBATCH --partition=nrnb-gpu
#SBATCH --account=nrnb-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --dependency=singleton


homedir="/cellar/users/cchuras/src/nest_vnn"
cn_deletions="${homedir}/sample/cell2cndeletion.txt"
cn_amplifications="${homedir}/sample/cell2cnamplification.txt"
gene2idfile="${homedir}/sample/gene2ind.txt"
cell2idfile="${homedir}/sample/cell2ind.txt"
ontfile="${homedir}/sample/ontology.txt"
mutationfile="${homedir}/sample/cell2mutation.txt"
traindatafile="${homedir}/sample/training_data.txt"

modeldir="${homedir}/sample/model"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

stdfile="${modeldir}/std.txt"
resultfile="${modeldir}/predict"

zscore_method="auc"

cudaid=0

pyScript="${homedir}/src/train_drugcell.py"

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -mutations $mutationfile -std $stdfile -modeldir $modeldir \
	-lr 0.0001 -wd 0.0001 -alpha 0.3 -cuda $cudaid -epoch 10 \
	-cn_deletions $cn_deletions -cn_amplifications $cn_amplifications \
	-batchsize 64 -optimize 1 -zscore_method $zscore_method > "${modeldir}/train.log"
