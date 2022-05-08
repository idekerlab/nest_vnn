import os
import numpy as np
import pandas as pd
import time
from scipy import stats
from multiprocessing import Pool
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)


class RLIPPCalculator():

	def __init__(self, args):
		self.ontology = pd.read_csv(args.ontology, sep='\t', header=None, names=['S', 'T', 'I'], dtype={0:str, 1:str, 2:str})
		self.terms = self.ontology['S'].unique().tolist()
		self.test_df = pd.read_csv(args.test, sep='\t', header=None, names=['C', 'D', 'AUC', 'DS'])
		self.predicted_vals = np.loadtxt(args.predicted)
		self.genes = pd.read_csv(args.gene2idfile, sep='\t', header=None, names=['I', 'G'])['G']
		self.cell_index = pd.read_csv(args.cell2idfile, sep="\t", header=None, names=['I', 'C'])
		self.rlipp_file = args.sys_output
		self.gene_rho_file = args.gene_output
		self.cpu_count = args.cpu_count
		self.num_hiddens_genotype = args.genotype_hiddens

		self.hidden_dir = args.hidden
		if not self.hidden_dir.endswith('/'):
			self.hidden_dir += '/'

		self.drugs = list(set(self.test_df['D']))
		self.drug_count = args.drug_count
		if self.drug_count == 0:
			self.drug_count = len(self.drugs)


	#Create a map of a list of the position of a drug in the test file
	def create_drug_pos_map(self):
		drug_pos_map = {d:[] for d in self.drugs}
		for i, row in self.test_df.iterrows():
			drug_pos_map[row['D']].append(i)
		return drug_pos_map


	# Create a sorted map of spearman correlation values for every drug
	def create_drug_corr_map_sorted(self, drug_pos_map):
		drug_corr_map = {}
		for d in self.drugs:
			if len(drug_pos_map[d]) == 0:
				drug_corr_map[d] = 0.0
				continue
			test_vals = np.take(np.array(self.test_df['AUC']), drug_pos_map[d])
			pred_vals = np.take(self.predicted_vals, drug_pos_map[d])
			drug_corr_map[d] = stats.spearmanr(test_vals, pred_vals)[0]
		return {drug:corr for drug,corr in sorted(drug_corr_map.items(), key=lambda item:item[1], reverse=True)}


	#Load the hidden file for a given element
	def load_feature(self, element, size):
		file_name = self.hidden_dir + element + '.hidden'
		return np.loadtxt(file_name, usecols=range(size))


	def load_term_features(self, term):
		return self.load_feature(term, self.num_hiddens_genotype)


	def load_gene_features(self, gene):
		return self.load_feature(gene, 1)


	def create_child_feature_map(self, feature_map, term):
		child_features = []
		child_features.append(term)
		children = [row['T'] for _,row in self.ontology.iterrows() if row['S']==term]
		for child in children:
			child_features.append(feature_map[child])
		return child_features


	#Load hidden features for all the terms and genes
	def load_all_features(self):
		feature_map = {}
		with Pool(self.cpu_count) as p:
			results = p.map(self.load_term_features, self.terms)
		for i,t in enumerate(self.terms):
			feature_map[t] = results[i]
		with Pool(self.cpu_count) as p:
			results = p.map(self.load_gene_features, self.genes)
		for i,g in enumerate(self.genes):
			feature_map[g] = results[i]

		child_feature_map = {t:[] for t in self.terms}
		for term in self.terms:
			children = [row['T'] for _,row in self.ontology.iterrows() if row['S']==term]
			for child in children:
				child_feature_map[term].append(feature_map[child])

		return feature_map, child_feature_map


	#Get a hidden feature matrix of a given term's children
	def get_child_features(self, term_child_features, position_map):
		child_features = []
		for f in term_child_features:
			child_features.append(np.take(f, position_map, axis=0))
		return np.column_stack([f for f in child_features])


	#Executes 5-fold cross validated Ridge regression for a given hidden features matrix
	#and returns the spearman correlation value of the predicted output
	def exec_lm(self, X, y):

		pca = PCA(n_components=self.num_hiddens_genotype)
		X_pca = pca.fit_transform(X)

		regr = RidgeCV(cv=5)
		regr.fit(X_pca, y)
		y_pred = regr.predict(X_pca)
		return stats.spearmanr(y_pred, y)


	# Calculates RLIPP for a given term and drug
	#Executes parallely
	def calc_term_rlipp(self, term_features, term_child_features, position_map, term, drug):
		X_parent = np.take(term_features, position_map, axis=0)
		X_child = self.get_child_features(term_child_features, position_map)
		y = np.take(self.predicted_vals, position_map)
		p_rho, p_pval = self.exec_lm(X_parent, y)
		c_rho, c_pval = self.exec_lm(X_child, y)
		rlipp = p_rho/c_rho
		result = '{}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\t{:.3e}\n'.format(term, p_rho, p_pval, c_rho, c_pval, rlipp)
		return result


	#Calculates Spearman correlation between Gene embeddings and Predicted AUC
	def calc_gene_rho(self, gene_features, position_map, gene, drug):
		pred = np.take(self.predicted_vals, position_map)
		gene_embeddings = np.take(gene_features, position_map)
		rho, p_val = stats.spearmanr(pred, gene_embeddings)
		result = '{}\t{:.3e}\t{:.3e}\n'.format(gene, rho, p_val)
		return result


	#Calculates RLIPP scores for top n drugs (n = drug_count), and
	#prints the result in "Drug Term P_rho C_rho RLIPP" format
	def calc_scores(self):
		print('Starting score calculation')

		drug_pos_map = self.create_drug_pos_map()
		sorted_drugs = list(self.create_drug_corr_map_sorted(drug_pos_map).keys())[0:self.drug_count]

		start = time.time()
		feature_map, child_feature_map = self.load_all_features()
		print('Time taken to load features: {:.4f}'.format(time.time() - start))

		rlipp_file = open(self.rlipp_file, "w")
		rlipp_file.write('Term\tP_rho\tP_pval\tC_rho\tC_pval\tRLIPP\n')
		gene_rho_file = open(self.gene_rho_file, "w")
		gene_rho_file.write('Gene\tRho\tP_val\n')

		with Parallel(backend="multiprocessing", n_jobs=self.cpu_count) as parallel:
			for i, drug in enumerate(sorted_drugs):
				start = time.time()

				rlipp_results = parallel(delayed(self.calc_term_rlipp)(feature_map[term], child_feature_map[term], drug_pos_map[drug], term, drug) for term in self.terms)
				for result in rlipp_results:
					rlipp_file.write(result)

				gene_rho_results = parallel(delayed(self.calc_gene_rho)(feature_map[gene], drug_pos_map[drug], gene, drug) for gene in self.genes)
				for result in gene_rho_results:
					gene_rho_file.write(result)

				print('Drug {} completed in {:.4f} seconds'.format((i+1), (time.time() - start)))
		gene_rho_file.close()
		rlipp_file.close()
