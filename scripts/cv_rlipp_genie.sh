#!/bin/bash

homedir=$1
ontology="${homedir}/data/training_files_av/ontology_${2}_${3}.txt"
gene2idfile="${homedir}/data/training_files_av/gene2ind_${2}_${3}.txt"
cell2idfile="${homedir}/data/GENIE/cell2ind_428.txt"
test="${homedir}/data/GENIE/test_428_${4}.txt"

modeldir="${homedir}/models/model_${2}_${3}_${4}_${5}_${6}"

predicted="${modeldir}/predict_genie_428.txt"
sys_output="${modeldir}/rlipp_genie_428.out"
gene_output="${modeldir}/gene_rho_genie_428.out"
hidden="${modeldir}/hidden_genie_428"

cpu_count=$7

genotype_hiddens=`grep "genotype_hiddens" "${modeldir}/train.log" | tail -1`
readarray -d : -t str <<< "$genotype_hiddens"
neurons=`echo "${str[1]}" | xargs`

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology -test $test \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -gene_output $gene_output \
	-predicted $predicted -cpu_count $cpu_count -drug_count 0 -genotype_hiddens $neurons > "${modeldir}/rlipp.log"
