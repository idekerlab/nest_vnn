#!/bin/bash

homedir=$1

ontology="${homedir}/data/training_files_${3}/ontology_${2}_${3}.txt"
gene2idfile="${homedir}/data/training_files_${3}/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/training_files_${3}/cell2ind_${3}.txt"
test="${homedir}/data/training_files_${3}/${6}_test_${3}_${4}.txt"

resultdir="${homedir}/models/shuffled_input/${6}_${7}"
predicted="${resultdir}/predict.txt"
sys_output="${resultdir}/rlipp.out"
gene_output="${resultdir}/gene_rho.out"
hidden="${resultdir}/hidden"

cpu_count=$8

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}_${6}"
genotype_hiddens=`grep "genotype_hiddens" "${modeldir}/train.log" | tail -1`
readarray -d : -t str <<< "$genotype_hiddens"
neurons=`echo "${str[1]}" | xargs`

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology -test $test \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -gene_output $gene_output \
	-predicted $predicted -cpu_count $cpu_count -drug_count 0 -genotype_hiddens $neurons > "${resultdir}/rlipp.log"
