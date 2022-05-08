#%%
'''QC Plots
- Function to generate QC plots of results from log files
'''
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def qcplots(filedir):
	with open(filedir + '/train.log', 'r') as fi:
		lines = fi.readlines()

	trialdict = {}
	done = False
	trial = 0
	for i, row in enumerate(lines):
		if 'genotype_hiddens' in row:
			trialend = i
			if trial > 0:
				data = lines[datastart:trialend]
				data = [line.strip('\n').split('\t') for line in data]
				header = data[0]
				data = data[1:]
				data = np.array([list(map(float,line)) for line in data])
				data = pd.DataFrame(data, columns = header)
				# Store
				trialdict[trial-1] = (trialinfo, data)
				trialinfo = None
				data = None
			trialstart = i
			trial += 1
			continue
		if 'epoch' in row:
			datastart = i
			trialinfo = lines[trialstart:datastart]
			continue
		if 'Study statistics:' in row:
			break

	trial_list = list(range(trial - 1))
	trial_batches = [trial_list[i:i + 8] for i in range(0, len(trial_list), 8)]
	plt.rcParams["figure.figsize"] = (8, 11)
	plt.rcParams["figure.autolayout"] = True

	batchcount = 1
	trialcounter = 0
	for batch in trial_batches:
		fig, axs = plt.subplots(len(batch), 3)
		for i, ax in enumerate(axs):
			# Extractdata
			trialparams = 'TRIAL_' + str(trialcounter) + '\n' + (',').join(trialdict[trialcounter][0])
			data = trialdict[trialcounter][1]
			# Make Figure
			ax[0].text(0, 0, trialparams)
			ax[0].axis('off')
			ax[1].plot(data.train_corr, label='Train Corr')
			ax[1].plot(data.val_corr, label='Val Corr')
			ax[2].plot(data.train_loss, label='Train Loss')
			ax[2].plot(data.val_loss, label='Val Loss')
			if i == 0:
				ax[2].legend()
			trialcounter += 1
		figname = (filedir + "/" + str(batchcount) + '_qcfig.pdf')
		plt.savefig(figname, dpi=300)
		batchcount += 1


if __name__=="__main__":
	qcplots(sys.argv[1])
