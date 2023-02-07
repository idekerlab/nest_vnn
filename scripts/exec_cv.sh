#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/dcodr"

dataset="av"
zscore_method="auc"
folds=5
nested_folds=1

#drugs=`awk '{ print $1 }' "${homedir}/data/training_files_av/drugname_${dataset}.txt"`
#drugs=`awk '{ print $1 }' "${homedir}/data/training_files_av/drugname_cmap.txt"`

for ont in "ctg"
do
	for drug in "Palbociclib" #"dichloroplatinum-diammoniate" "125316-60-1" "camptothecin" "Ceralasertib"
	do
		for ((i=1;i<=folds;i++));
		do
			for ((j=1;j<=nested_folds;j++));
			do
				sbatch -J "DCoDR_${ont}_${drug}_${i}" -o "${homedir}/logs/out_${ont}_${drug}_${i}.log" ${homedir}/scripts/batch_cv.sh $homedir $ont $dataset $drug ${zscore_method} $i $j
				sbatch -J "DCoDR_${ont}_${drug}_${i}" -o "${homedir}/logs/rlipp_${ont}_${drug}_${i}.log" ${homedir}/scripts/rlipp_slurm_cv.sh $homedir $ont $dataset $drug ${zscore_method} $i $j
			done
		done
	done
done