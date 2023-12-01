#!/bin/bash
homedir="$1"

gene2idfile="${homedir}/sample/gene2ind.txt"
cell2idfile="${homedir}/sample/cell2ind.txt"
ontfile="${homedir}/sample/ontology.txt"
mutationfile="${homedir}/sample/cell2mutation.txt"
cn_deletionfile="${homedir}/sample/cell2cndeletion.txt"
cn_amplificationfile="${homedir}/sample/cell2cnamplification.txt"
traindatafile="${homedir}/sample/training_data.txt"

modeldir="${homedir}/model"
if [ -d $modeldir ]
then
	rm -rf $modeldir
fi
mkdir -p $modeldir

stdfile="${modeldir}/std.txt"
resultfile="${modeldir}/predict"

zscore_method="auc"

cudaid=0

pyScript="${homedir}/src/train.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile -train $traindatafile \
	-mutations $mutationfile -cn_deletions $cn_deletionfile -cn_amplifications $cn_amplificationfile \
	-std $stdfile -model $modeldir -genotype_hiddens 4 -lr 0.0005 -cuda $cudaid -epoch 50 \
	-batchsize 64 -optimize 1 -zscore_method $zscore_method > "${modeldir}/train.log"
