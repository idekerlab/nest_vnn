#!/bin/bash

homedir=$1
col_num=$2
output_file=$3

awk -v col=$col_num 'BEGIN {FS="\t"} {print $col}' "$homedir/data/NCI/nci_auc.tab" | awk 'NR!=1 {print}' | sort -u | awk 'BEGIN {OFS="\t"} {print NR-1, $1}' > $output_file