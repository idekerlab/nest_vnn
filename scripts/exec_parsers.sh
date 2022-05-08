#!/bin/bash

homedir="/cellar/users/asinghal/Workspace/dcodr"

#Drugs

bash "$homedir/scripts/parse_ctrp_gdsc_auc_files.sh" $homedir 2 "$homedir/data/training_files_av/drug_list_av.txt" "$homedir/data/training_files_av/drug2ind_av.txt"

#bash "$homedir/scripts/parse_nci_auc_file.sh" $homedir 4 "$homedir/data/drug2ind_nci.txt"


#Cell lines

bash "$homedir/scripts/parse_ctrp_gdsc_auc_files.sh" $homedir 1 "$homedir/data/training_files_av/cell_list_av.txt" "$homedir/data/training_files_av/cell2ind_av.txt"

#bash "$homedir/scripts/parse_nci_auc_file.sh" $homedir 5 "$homedir/data/cell2ind_nci.txt"
