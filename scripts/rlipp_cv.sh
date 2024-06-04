#!/bin/bash

homedir=$HOME/nest_vnn

ontology="${homedir}/sample/ontology.txt"
gene2idfile="${homedir}/sample/gene2ind.txt"
cell2idfile="${homedir}/sample/cell2ind.txt"
test="${homedir}/sample/test_data.txt"

#i=$6
#j=$7
#nf=$(( i + 5*(j-1) ))
modeldir="${homedir}/pretrained_models/palbociclib/model_${2}_${3}_${4}_${5}"

predicted="${modeldir}/my_predict.txt"
sys_output="${modeldir}/my_rlipp.out"
gene_output="${modeldir}/my_gene_rho.out"
hidden="${modeldir}/hidden"

cpu_count=$8

#genotype_hiddens=`grep "genotype_hiddens" "${modeldir}/train.log" | tail -1`
#readarray -d : -t str <<< "$genotype_hiddens"
#neurons=`echo "${str[1]}" | xargs`
neurons=4

python -u ${homedir}/src/rlipp_helper.py -hidden $hidden -ontology $ontology -test $test \
	-gene2idfile $gene2idfile -cell2idfile $cell2idfile -sys_output $sys_output -gene_output $gene_output \
	-predicted $predicted -cpu_count $cpu_count -drug_count 0 -genotype_hiddens $neurons > "${modeldir}/my_rlipp.log"
