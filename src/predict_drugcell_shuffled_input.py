
import argparse
import sys
import os
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import util


def predict_drugcell(predict_data, gene_dim, model_file, hidden_folder, batch_size, result_file, cell_features):

	feature_dim = gene_dim

	model = torch.load(model_file, map_location='cuda:%d' % CUDA_ID)

	predict_feature, predict_label = predict_data

	predict_label_gpu = predict_label.cuda(CUDA_ID)

	model.cuda(CUDA_ID)
	model.eval()

	test_loader = du.DataLoader(du.TensorDataset(predict_feature, predict_label), batch_size=batch_size, shuffle=False)

	#Test
	test_predict = torch.zeros(0,0).cuda(CUDA_ID)
	hidden_embeddings_map = {}

	saved_grads = {}
	def save_grad(element):
		def savegrad_hook(grad):
			saved_grads[element] = grad
		return savegrad_hook

	for i, (inputdata, labels) in enumerate(test_loader):
		# Convert torch tensor to Variable
		features = util.build_input_vector(inputdata, cell_features)

		cuda_features = Variable(features.cuda(CUDA_ID), requires_grad=True)

		# make prediction for test data
		aux_out_map, hidden_embeddings_map = model(cuda_features)

		if test_predict.size()[0] == 0:
			test_predict = aux_out_map['final'].data
		else:
			test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

		for element, hidden_map in hidden_embeddings_map.items():
			hidden_file = hidden_folder + '/' + element + '.hidden'
			with open(hidden_file, 'ab') as f:
				np.savetxt(f, hidden_map.data.cpu().numpy(), '%.4e')

		for element, _ in hidden_embeddings_map.items():
			hidden_embeddings_map[element].register_hook(save_grad(element))

		## Do backprop
		aux_out_map['final'].backward(torch.ones_like(aux_out_map['final']))

		# Save Feature Grads
		feature_grad = torch.zeros(0,0).cuda(CUDA_ID)
		for i in range(len(cuda_features[0, 0, :])):
			feature_grad = cuda_features.grad.data[:, :, i]
			with open(result_file + '_feature_grad_' + str(i) + '.txt', 'ab') as f:
				np.savetxt(f, feature_grad.cpu().numpy(), '%.4e', delimiter='\t')

		# Save Hidden Grads
		for element, hidden_grad in saved_grads.items():
			hidden_file = hidden_folder + '/' + element + '.hidden_grad'
			with open(hidden_file, 'ab') as f:
				np.savetxt(f, hidden_grad.data.cpu().numpy(), '%.4e', delimiter='\t')

	test_corr = util.pearson_corr(test_predict, predict_label_gpu)
	print("Test correlation\t%s\t%.4f" % (model.root, test_corr))

	np.savetxt(result_file + '.txt', test_predict.cpu().numpy(),'%.4e')


parser = argparse.ArgumentParser(description='Predict DrugCell')
parser.add_argument('-predict', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=1000)
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str)
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str)
parser.add_argument('-load', help='Model file', type=str)
parser.add_argument('-hidden', help='Hidden output folder', type=str, default='hidden/')
parser.add_argument('-result', help='Result file prefix', type=str, default='result/predict')
parser.add_argument('-cuda', help='Specify GPU', type=int, default=0)
parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
parser.add_argument('-zscore_method', help='zscore method (zscore/robustz)', type=str)
parser.add_argument('-std', help = 'Standardization File', type = str)

opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_data, cell2id_mapping = util.prepare_predict_data(opt.predict, opt.cell2id, opt.zscore_method, opt.std)
gene2id_mapping = util.load_mapping(opt.gene2id, "genes")

# load cell/drug features
mutations = np.genfromtxt(opt.mutations, delimiter = ',')
temp = np.transpose(mutations)
np.random.shuffle(temp)
mutations = np.transpose(temp)

cn_deletions = np.genfromtxt(opt.cn_deletions, delimiter = ',')
temp = np.transpose(cn_deletions)
np.random.shuffle(temp)
cn_deletions = np.transpose(temp)

cn_amplifications = np.genfromtxt(opt.cn_amplifications, delimiter = ',')
temp = np.transpose(cn_amplifications)
np.random.shuffle(temp)
cn_amplifications = np.transpose(temp)

cell_features = np.dstack([mutations, cn_deletions, cn_amplifications])

num_cells = len(cell2id_mapping)
num_genes = len(gene2id_mapping)

CUDA_ID = opt.cuda

predict_drugcell(predict_data, num_genes, opt.load, opt.hidden, opt.batchsize, opt.result, cell_features)
