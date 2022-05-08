import sys
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag

import util


class TrainingDataWrapper():

	def __init__(self, args):

		self.cell_id_mapping = util.load_mapping(args.cell2id, 'cell lines')
		self.gene_id_mapping = util.load_mapping(args.gene2id, 'genes')
		self.num_hiddens_genotype = args.genotype_hiddens
		self.lr = args.lr
		self.wd = args.wd
		self.alpha = args.alpha
		self.epochs = args.epoch
		self.batchsize = args.batchsize
		self.modeldir = args.modeldir
		self.cuda = args.cuda
		self.train = args.train
		self.zscore_method = args.zscore_method
		self.std = args.std
		self.patience = args.patience
		self.delta = args.delta
		self.min_dropout_layer = args.min_dropout_layer
		self.dropout_fraction = args.dropout_fraction
		self.load_ontology(args.onto)

		self.mutations = np.genfromtxt(args.mutations, delimiter = ',')
		self.cn_deletions = np.genfromtxt(args.cn_deletions, delimiter = ',')
		self.cn_amplifications = np.genfromtxt(args.cn_amplifications, delimiter = ',')
		self.cell_features = np.dstack([self.mutations, self.cn_deletions, self.cn_amplifications])

		self.train_feature, self.train_label, self.val_feature, self.val_label = self.prepare_train_data()


	def prepare_train_data(self):
		return util.prepare_train_data(self.train, self.cell_id_mapping, self.zscore_method, self.std)

	def load_ontology(self, file_name):

		dG = nx.DiGraph()
		term_direct_gene_map = {}
		term_size_map = {}
		gene_set = set()

		file_handle = open(file_name)
		for line in file_handle:
			line = line.rstrip().split()
			if line[2] == 'default':
				dG.add_edge(line[0], line[1])
			else:
				if line[1] not in self.gene_id_mapping:
					continue
				if line[0] not in term_direct_gene_map:
					term_direct_gene_map[line[0]] = set()
				term_direct_gene_map[line[0]].add(self.gene_id_mapping[line[1]])
				gene_set.add(line[1])
		file_handle.close()

		for term in dG.nodes():
			term_gene_set = set()
			if term in term_direct_gene_map:
				term_gene_set = term_direct_gene_map[term]
			deslist = nxadag.descendants(dG, term)
			for child in deslist:
				if child in term_direct_gene_map:
					term_gene_set = term_gene_set | term_direct_gene_map[child]
			# jisoo
			if len(term_gene_set) == 0:
				print('There is empty terms, please delete term:', term)
				sys.exit(1)
			else:
				term_size_map[term] = len(term_gene_set)

		roots = [n for n in dG.nodes if dG.in_degree(n) == 0]

		uG = dG.to_undirected()
		connected_subG_list = list(nxacc.connected_components(uG))

		print('There are', len(roots), 'roots:', roots[0])
		print('There are', len(dG.nodes()), 'terms')
		print('There are', len(connected_subG_list), 'connected componenets')

		if len(roots) > 1:
			print('There are more than 1 root of ontology. Please use only one root.')
			sys.exit(1)
		if len(connected_subG_list) > 1:
			print('There are more than connected components. Please connect them.')
			sys.exit(1)

		self.dG = dG
		self.root = roots[0]
		self.term_size_map = term_size_map
		self.term_direct_gene_map = term_direct_gene_map
