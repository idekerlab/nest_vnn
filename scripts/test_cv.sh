#!/bin/bash
homedir=$1
zscore_method=$5

gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"
mutationfile="${homedir}/data/training_files_${3}/cell2mutation_${2}_${3}.txt"
cn_deletionfile="${homedir}/data/training_files_${3}/cell2cndeletion_${2}_${3}.txt"
cn_amplificationfile="${homedir}/data/training_files_${3}/cell2cnamplification_${2}_${3}.txt"
testdatafile="${homedir}/data/training_files_${3}/${6}_test_${3}_${4}.txt"

i=$6
j=$7
nf=$(( i + 5*(j-1) ))
modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}_${nf}"
modelfile="${modeldir}/model_final.pt"

stdfile="${modeldir}/std.txt"

resultfile="${modeldir}/predict"

hiddendir="${modeldir}/hidden"
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
