#!/bin/bash
homedir=$1
zscore_method=$5

gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/GENIE/cell2ind_428.txt"
mutationfile="${homedir}/data/GENIE/cell2mutation_428.txt"
cn_deletionfile="${homedir}/data/GENIE/cell2cndeletion_428.txt"
cn_amplificationfile="${homedir}/data/GENIE/cell2cnamplification_428.txt"
testdatafile="${homedir}/data/GENIE/test_428_${4}.txt"

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}_${6}"
modelfile="${modeldir}/model_final.pt"

stdfile="${modeldir}/std.txt"

resultfile="${modeldir}/predict_genie_428"

hiddendir="${modeldir}/hidden_genie_428"
if [ -d $hiddendir ]
then
	rm -rf $hiddendir
fi
mkdir -p $hiddendir

cudaid=0

pyScript="${homedir}/src/predict_drugcell.py"

source activate cuda11_env

python -u $pyScript -gene2id $gene2idfile -cell2id $cell2idfile -std $stdfile -hidden $hiddendir -result $resultfile \
	-mutations $mutationfile -cn_deletions $cn_deletionfile -cn_amplifications $cn_amplificationfile \
	-batchsize 2000 -predict $testdatafile -zscore_method $zscore_method -load $modelfile -cuda $cudaid > "${modeldir}/test.log"
