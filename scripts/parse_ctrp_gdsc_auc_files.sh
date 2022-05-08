#!/bin/bash

homedir=$1
col_num=$2
temp_file=$3
output_file=$4

# Get all the entries in col_num from all the files except line 1
awk -v col=$col_num 'BEGIN {FS="\t"} {print $col}' "$homedir/data/GDSC/gdsc2_auc_av.tab" | awk 'NR!=1 {print}' > $temp_file
awk -v col=$col_num 'BEGIN {FS="\t"} {print $col}' "$homedir/data/CTRP/ctrp1_auc_av.tab" | awk 'NR!=1 {print}' >> $temp_file
awk -v col=$col_num 'BEGIN {FS="\t"} {print $col}' "$homedir/data/CTRP/ctrp2_auc_av.tab" | awk 'NR!=1 {print}' >> $temp_file
awk -v col=$col_num 'BEGIN {FS="\t"} {print $col}' "$homedir/data/GDSC/gdsc1_auc_av.tab" | awk 'NR!=1 {print}' >> $temp_file

#Create the index file
sed '/^$/d' $temp_file | sort -u | awk 'BEGIN {OFS="\t"} {print NR-1, $1}' > $output_file