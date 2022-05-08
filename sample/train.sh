#!/bin/bash

homedir="your_home_directory"

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

pyScript="${homedir}/src/train.py"

source activate cuda11_env

python -u $pyScript -onto $ontfile -gene2id $gene2idfile -cell2id $cell2idfile \
	-train $traindatafile -genotype $mutationfile -std $stdfile -model $modeldir \
	-genotype_hiddens 6 -lr 0.0001 -wd 0.0001 -alpha 0.3 -cuda $cudaid -epoch 300 \
	-batchsize 64 -optimize 1 -zscore_method $zscore_method > "${modeldir}/train.log"
