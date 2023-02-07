import argparse
import copy

from nn_trainer import *
from optuna_nn_trainer import *
from gradient_nn_trainer import *


def main():

	torch.set_printoptions(precision = 5)

	parser = argparse.ArgumentParser(description = 'Train DrugCell')
	parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	parser.add_argument('-train', help = 'Training dataset', type = str)
	parser.add_argument('-epoch', help = 'Training epochs for training', type = int, default = 300)
	parser.add_argument('-lr', help = 'Learning rate', type = float, default = 0.001)
	parser.add_argument('-wd', help = 'Weight decay', type = float, default = 0.001)
	parser.add_argument('-alpha', help = 'Loss parameter alpha', type = float, default = 0.3)
	parser.add_argument('-batchsize', help = 'Batchsize', type = int, default = 64)
	parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'MODEL/')
	parser.add_argument('-cuda', help = 'Specify GPU', type = int, default = 0)
	parser.add_argument('-gene2id', help = 'Gene to ID mapping file', type = str)
	parser.add_argument('-cell2id', help = 'Cell to ID mapping file', type = str)
	parser.add_argument('-genotype_hiddens', help = 'Mapping for the number of neurons in each term in genotype parts', type = int, default = 4)
	parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
	parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
	parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
	parser.add_argument('-optimize', help = 'Hyper-parameter optimization', type = int, default = 1)
	parser.add_argument('-zscore_method', help='zscore method (zscore/robustz)', type=str, default = 'auc')
	parser.add_argument('-std', help = 'Standardization File', type = str, default = 'MODEL/std.txt')
	parser.add_argument('-patience', help = 'Early stopping epoch limit', type = int, default = 30)
	parser.add_argument('-delta', help = 'Minimum change in loss to be considered an improvement', type = float, default = 0.001)
	parser.add_argument('-min_dropout_layer', help = 'Start dropout from this Layer number', type = int, default = 2)
	parser.add_argument('-dropout_fraction', help = 'Dropout Fraction', type = float, default = 0.3)

	opt = parser.parse_args()
	data_wrapper = TrainingDataWrapper(opt)

	if opt.optimize == 0:
		NNTrainer(data_wrapper).train_model()

	elif opt.optimize == 1:
		GradientNNTrainer(data_wrapper).train_model()

	elif opt.optimize == 2:
		trial_params = OptunaNNTrainer(data_wrapper).exec_study()
		for key, value in trial_params.items():
			if hasattr(data_wrapper, key):
				setattr(data_wrapper, key, value)
		GradientNNTrainer(data_wrapper).train_model()

	else:
		print("Wrong value for optimize.")
		exit(1)

if __name__ == "__main__":
	main()
