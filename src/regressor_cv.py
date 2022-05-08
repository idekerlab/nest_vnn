'''Regresor  Gridsearch CV

Purpose: to perform regression gridsearch using a variety of models, select best model, and test performance on GENIE data

Input:

Output:

Erica Silva'''
# %%
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GroupKFold
import os

class gridsearch() :

	def __init__(self, args):
		self.datadir = args.datadir
		self.nfolds = args.nfolds
		self.drug = args.drug
		self.predict_clinical = args.predict_clinical
		if self.predict_clinical == 'yes':
			self.clinicaldatadir = args.clinicaldatadir
		self.resultdir = args.resultdir
		self.method = args.method
		self.dataset = 'all'

		print('Running Regressor Gridsearch with the following options')
		print(args)
		self.get_folds()
		self.prep_data() # Prep traintestval data
		# CV
		if self.method == 'elasticnet':
			self.elasticnet()
		elif self.method == 'mlp':
			self.mlp()
		else:
			print('Classifier method not specified. Please revise')
		self.findparams() # Do CV

	def get_folds(self):
		print('\tGetting Data Directories')
		count = 1
		folddict = {}
		for fold in range(1, self.nfolds+1): # Iterate over folds
			datadict = {}
			# Save locations of test and train datasets
			datadict['traindir'] = self.datadir + '/' + str(fold) + '_train_av_' + self.drug + '.txt'
			datadict['testdir'] = self.datadir + '/' + str(fold) + '_test_av_' + self.drug + '.txt'
			for label in ['train_x', 'train_y', 'test_x', 'test_y']:
				datadict[label] = '/' + str(count) + '.' + label + '.npy'
			folddict[count] = datadict
			count += 1

		if self.predict_clinical == 'yes':
			datadict = {}
			datadict['clinicaldir'] = self.clinicaldatadir + '/GENIE_test_av_' + self.drug + '.txt'
			for label in ['train_x', 'train_y']:
				datadict[label] = '/' + str(count) + '.clinical.' + label + '.npy'
			folddict['clinical'] = datadict
		self.folds = folddict

	def prep_data(self):
		if self.dataset =='genie':
			fold = self.folds['clinical']
			clinicaldata = pd.read_csv(fold['clinicaldir'], header=None, delimiter='\t')
			train_x, train_y, _ , _ = self.format_data(clinicaldata, None)
			np.save(self.fi + fold['train_x'], train_x)
			np.save(self.fi + fold['train_y'], train_y)
			self.genie_x = train_x
			self.genie_y = train_y
			self.dataset == 'all'
		else:
			self.fi = self.datadir + '/' + self.drug
			if os.path.exists(self.fi):
				import shutil
				shutil.rmtree(self.fi)
			os.mkdir(self.fi)
			# Iterate over folds
			kf = GroupKFold(n_splits=self.nfolds)
			for f in range(1, self.nfolds+1):
				# First work on training data
				fold = self.folds[f]
				train_df = pd.read_csv(fold['traindir'], header=None, delimiter='\t')
				# Get indices for train and val subsets of train dataset
				i = np.random.randint(1,high=self.nfolds+1)
				for g, (train_ind, val_ind) in enumerate(kf.split(train_df, groups=train_df[0].values)):
					if g+1 == i:
						self.folds[f]['train_ind'] = train_ind
						self.folds[f]['val_ind'] = val_ind
						break
				test_df = pd.read_csv(fold['testdir'], header=None, delimiter='\t')
				# Format Folds
				train_x, train_y, test_x, test_y = self.format_data(train_df, test_df)

				# Save Data
				np.save(self.fi + fold['train_x'], train_x)
				np.save(self.fi + fold['train_y'], train_y)
				np.save(self.fi + fold['test_x'], test_x)
				np.save(self.fi + fold['test_y'], test_y)

			del train_x, train_y, test_x, test_y
			if self.predict_clinical == 'yes':
				self.dataset = 'genie'
		print(f'\tData formatted and saved at {self.fi}')

	def format_data(self, train_df, val_df):
		# Load necessary data
		gene_index = pd.read_csv(self.datadir + '/gene2ind_ctg_av.txt', sep='\t', header=None, names=(['I', 'G']))
		gene_list = gene_index['G']

		if self.dataset == 'all':
			cell2inddir = self.datadir + '/cell2ind_av.txt'
			cell2mutdir = self.datadir + '/cell2mutation_ctg_av.txt'
		elif self.dataset == 'genie':
			cell2inddir = self.clinicaldatadir + '/GENIE_all_cell2ind.txt'
			cell2mutdir = self.clinicaldatadir + '/GENIE_cell2mutation_av.txt'

		cell_index = pd.read_csv(cell2inddir, sep='\t', header=None, names=(['I', 'C']))
		cell_map = dict(zip(cell_index['C'], cell_index['I']))
		cell_features = pd.read_csv(cell2mutdir, header=None, names=gene_list)

		train_df.columns = ['cell', 'drug', 'auc', 'dataset']
		train_Y = np.array(train_df['auc'])
		train_X = np.empty(shape = (len(train_df), len(gene_list))) # Convert format X for training
		train_X[:] = np.nan
		for i, row in enumerate(train_df.values):
			train_X[i,:] = cell_features.iloc[int(cell_map[row[0]])]
		if np.isnan(train_X).any():
			print('There are nan in the Train X data')
			exit()

		if val_df is not None:
			val_df.columns = ['cell', 'drug', 'auc', 'dataset']
			test_y = np.array(val_df['auc'])
			test_x = np.empty(shape = (len(val_df), len(gene_list)))
			test_x[:] = np.nan
			for i, row in enumerate(val_df.values):
				test_x[i,:] = cell_features.iloc[int(cell_map[row[0]])]
			if np.isnan(test_x).any():
				print('There are nan in the Test X data')
				exit()
		else:
			test_x = None
			test_y = None

		return train_X, train_Y, test_x, test_y

	def elasticnet(self):
		print('\tSelecting ElasticNet')
		paramgrid = {'l1_ratio' : [.1, .5, .7, .9, .95, .99, 1], 'alpha' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]}

		regr = ElasticNet(max_iter=1000)
		self.paramgrid = paramgrid
		self.regr = regr

	def mlp(self):
		print('\tSelecting MLP Regression')
		# paramgrid = {'hidden_layer_sizes' : [(30, 84, 150, 240, 258, 18,6)], 'alpha' : [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0], 'learning_rate': ['constant', 'adaptive'], 'batch_size' : [32, 64, 96, 128]}
		paramgrid = {'hidden_layer_sizes' : [(100,50,6), (100)], 'alpha' : [1e-4, 1e-5], 'learning_rate': ['constant'], 'batch_size' : [32]} # for debug

		regr = regr = MLPRegressor(activation ='relu', shuffle=True, early_stopping=True)
		self.paramgrid = paramgrid
		self.regr = regr

	def findparams(self):

		# Track what's happening
		print(f'Number of folds for CV:\t{self.nfolds}')
		print('Scanning params:')
		print(self.paramgrid)

		resultnames = ['Train Corr', 'Val Corr']
		cols = list(self.paramgrid.keys()) + resultnames
		output = pd.DataFrame(columns = cols)

		for fold in range(1, self.nfolds+1):
			print(f'Beginning fold {fold}')
			# Train & Val
			for paramset in ParameterGrid(self.paramgrid):
				# Load Data
				regr = self.regr.set_params(**paramset)
				train_ind = self.folds[fold]['train_ind']
				val_ind = self.folds[fold]['val_ind']
				train_x = np.load(self.fi + self.folds[fold]['train_x'])
				train_y = np.load(self.fi + self.folds[fold]['train_y'])
				test_x = np.load(self.fi + self.folds[fold]['test_x'])
				# test_y =  np.load(self.fi + self.folds[fold]['test_y'])
				# Fit
				regr.fit(train_x[train_ind,:], train_y[train_ind])
				# Training Predictions
				result = paramset.copy()
				result['Train Corr'] = stats.pearsonr(train_y[train_ind], regr.predict(train_x[train_ind,:]))[0]
				result['Val Corr'] = stats.pearsonr(train_y[val_ind], regr.predict(train_x[val_ind,:]))[0]
				output = output.append(result, ignore_index=True)

			output.sort_values(by='Val Corr', ascending=False, inplace=True)
			output.reset_index(inplace=True, drop=True)
			testparams = output.loc[0,paramset.keys()].to_dict()
			print(f'\tSelecting best paramset: {testparams}')
			regr = self.regr.set_params(**testparams)
			# Best params train and test
			regr.fit(train_x, train_y)
			predict_y = regr.predict(test_x)
			# Save Data
			doutdir = self.resultdir + '/' + str(fold) + '_' + self.drug
			if os.path.exists(doutdir):
				import shutil
				shutil.rmtree(doutdir)
			os.mkdir(doutdir)
			output.to_csv(doutdir+'/cv_log.txt')
			np.savetxt(doutdir+'/predict.txt', predict_y)
			if self.predict_clinical == 'yes':
				clinical_x = np.load(self.fi + self.folds['clinical']['train_x'])
				predict_clinical = regr.predict(clinical_x)
				np.savetxt(doutdir+'/geniepredict.txt', predict_clinical)
			print(f'\t...Finished Fold')

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description = 'Train DrugCell')
	parser.add_argument('-datadir', help = 'Directory where train, test, cell2ind, cell2mutation, gene2ind data can be found', type=str, default='/cellar/shared/asinghal/nest_drugcell/data/training_files_av')
	parser.add_argument('-drug', help = 'drugname to process', type=str, default = 'Palbociclib')
	parser.add_argument('-nfolds', help= 'Number of folds for CV', type = int, default = 5)
	parser.add_argument('-resultdir', help = 'Directory where results should be stored', type = str, default='/cellar/users/e5silva/Software/nest_drugcell/es_scripts/test')
	parser.add_argument('-method', help = 'The type of regression: "elasticnet" or "mlp"', type = str, default = 'mlp')
	parser.add_argument('-predict_clinical', help = 'Whether or not to predict clinical trial data. Options: "yes", "no".', type=str, default="no")
	parser.add_argument('-clinicaldatadir', help = 'Directory where clinical trial data can be found', type=str, default='/cellar/shared/asinghal/nest_drugcell/data/GENIE')
	opt = parser.parse_args()
	gridsearch(opt)
# %%
