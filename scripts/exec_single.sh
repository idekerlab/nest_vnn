#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/dcodr"

dataset="av"
zscore_method="auc"

#drugs=`awk '{ print $1 }' "${homedir}/data/training_files_av/drugname_${dataset}.txt"`
#drugs=`awk '{ print $1 }' "${homedir}/data/training_files_av/drugname_cmap.txt"`

for ont in "ctg"
do
	for drug in "gemcitabine" #"dichloroplatinum-diammoniate" "125316-60-1" "camptothecin" "Ceralasertib"
	do
		sbatch -J "DCoDR_${ont}_${drug}" -o "${homedir}/logs/out_${ont}_${drug}.log" ${homedir}/scripts/batch.sh $homedir $ont $dataset $drug ${zscore_method}
		sbatch -J "DCoDR_${ont}_${drug}" -o "${homedir}/logs/rlipp_${ont}_${drug}.log" ${homedir}/scripts/rlipp_slurm.sh $homedir $ont $dataset $drug ${zscore_method}
	done
done
