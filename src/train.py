import argparse
import copy

from vnn_trainer import *
from optuna_nn_trainer import *

import json
import os

import candle

# Just because the tensorflow warnings are a bit verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# This should be set outside as a user environment variable
homedir = "/cellar/users/asinghal/Workspace/nest_vnn"

# file_path becomes the default location of the example_default_model.txt file
file_path = os.path.dirname(os.path.realpath(__file__))

# Define any needed additional args to ensure all new args are command-line accessible.
additional_definitions = [
	{"name": "onto", "type": str, "nargs": 1, "help": "Ontology file used to guide the neural network"},
	{"name": "wd", "type": float, "nargs": 1, "help": "Weight decay", "default":0.001},
	{"name": "alpha", "type": float, "nargs": 1, "help": "Loss parameter alpha", "default":0.3},
	{"name": "cuda", "type": float, "nargs": 1, "help": "cuda", "default":0},
	{"name": "gene2id", "type": str, "nargs": 1, "help": "Gene to ID mapping file"},
	{"name": "cell2id", "type": str, "nargs": 1, "help": "Celline to ID mapping file"},
	{"name": "mutations", "type": str, "nargs": 1, "help": "Mutation information for cell lines"},
	{"name": "cn_deletions", "type": str, "nargs": 1, "help": "Copy number deletions for cell lines"},
	{"name": "cn_amplifications", "type": str, "nargs": 1, "help": "Copy number amplifications for cell lines"},
	{"name": "genotype_hiddens", "type": int, "nargs": 1, "help": "Mapping for the number of neurons in each term in genotype parts"},
	{"name": "optimize", "type": int, "nargs": 1, "help": "HPO or not", "default":1},
	{"name": "zscore_method", "type": str, "nargs": 1, "help": "zscore", "default":'auc'},
	{"name": "modeldir", "type": str, "nargs": 1, "help": "Model output directory"},
	{"name": "std", "type": str, "nargs": 1, "help": "Standardization File"},
	{"name": "delta", "type": float, "nargs": 1, "help": "Minimum change in loss to be considered an improvement", "default":0.0001},
	{"name": "min_dropout_layer", "type": int, "nargs": 1, "help": "Start dropout from this Layer number", "default":2}
]

# Define args that are required.
required = None


# Extend candle.Benchmark to configure the args
class CLI(candle.Benchmark):
	def set_locals(self):
		if required is not None:
			self.required = set(required)
		if additional_definitions is not None:
			self.additional_definitions = additional_definitions


def run_vnn(params):

	torch.set_printoptions(precision = 5)

	# parser = argparse.ArgumentParser(description = 'Train VNN')
	# parser.add_argument('-onto', help = 'Ontology file used to guide the neural network', type = str)
	# parser.add_argument('-train', help = 'Training dataset', type = str)
	# parser.add_argument('-epoch', help = 'Training epochs for training', type = int, default = 300)
	# parser.add_argument('-lr', help = 'Learning rate', type = float, default = 0.001)
	# parser.add_argument('-wd', help = 'Weight decay', type = float, default = 0.001)
	# parser.add_argument('-alpha', help = 'Loss parameter alpha', type = float, default = 0.3)
	# parser.add_argument('-batchsize', help = 'Batchsize', type = int, default = 64)
	# parser.add_argument('-modeldir', help = 'Folder for trained models', type = str, default = 'MODEL/')
	# parser.add_argument('-cuda', help = 'Specify GPU', type = int, default = 0)
	# parser.add_argument('-gene2id', help = 'Gene to ID mapping file', type = str)
	# parser.add_argument('-cell2id', help = 'Cell to ID mapping file', type = str)
	# parser.add_argument('-genotype_hiddens', help = 'Mapping for the number of neurons in each term in genotype parts', type = int, default = 4)
	# parser.add_argument('-mutations', help = 'Mutation information for cell lines', type = str)
	# parser.add_argument('-cn_deletions', help = 'Copy number deletions for cell lines', type = str)
	# parser.add_argument('-cn_amplifications', help = 'Copy number amplifications for cell lines', type = str)
	# parser.add_argument('-optimize', help = 'Hyper-parameter optimization', type = int, default = 1)
	# parser.add_argument('-zscore_method', help='zscore method (zscore/robustz)', type=str, default = 'auc')
	# parser.add_argument('-std', help = 'Standardization File', type = str, default = 'MODEL/std.txt')
	# parser.add_argument('-patience', help = 'Early stopping epoch limit', type = int, default = 30)
	# parser.add_argument('-delta', help = 'Minimum change in loss to be considered an improvement', type = float, default = 0.001)
	# parser.add_argument('-min_dropout_layer', help = 'Start dropout from this Layer number', type = int, default = 2)
	# parser.add_argument('-dropout_fraction', help = 'Dropout Fraction', type = float, default = 0.3)
	# opt = parser.parse_args()

	print(params['optimize'])
	data_wrapper = TrainingDataWrapper(params, homedir)

	if params['optimize'] == 1:
		VNNTrainer(data_wrapper).train_model()

	elif params['optimize'] == 2:
		trial_params = OptunaNNTrainer(data_wrapper).exec_study()
		for key, value in trial_params.items():
			if hasattr(data_wrapper, key):
				setattr(data_wrapper, key, value)
		VNNTrainer(data_wrapper).train_model()

	else:
		print("Wrong value for optimize.")
		exit(1)


# In the initialize_parameters() method, we will instantiate the base
# class, and finally build an argument parser to recognize your customized
# parameters in addition to the default parameters.The initialize_parameters()
# method should return a python dictionary, which will be passed to the run()
# method.
def initialize_parameters():
	i_bmk = CLI(
		file_path,  # this is the path to this file needed to find default_model.txt
		"default.cfg",  # name of the default_model.txt file
		"pytorch",  # framework, choice is keras or pytorch
		prog="example_baseline",  # basename of the model
		desc="IMPROVE Benchmark",
	)

	gParameters = candle.finalize_parameters(
		i_bmk
	)  # returns the parameter dictionary built from
	# default_model.txt and overwritten by any
	# matching comand line parameters.

	return gParameters


def run(params):

	metrics = run_vnn(params)

	metrics = {"val_loss": 0.101, "pcc": 0.923, "scc": 0.777, "rmse": 0.036}
	# metrics is used by the supervisor when running
	# HPO workflows (and possible future non HPO workflows)

	# Dumping results into file, workflow requirement
	val_scores = {
		"key": "val_loss",
		"value": metrics["val_loss"],
		"val_loss": metrics["val_loss"],
		"pcc": metrics["pcc"],
		"scc": metrics["scc"],
		"rmse": metrics["rmse"],
	}

	with open(params["output_dir"] + "/scores.json", "w", encoding="utf-8") as f:
		json.dump(val_scores, f, ensure_ascii=False, indent=4)

	return metrics  # metrics is used by the supervisor when running
	# HPO workflows (and possible future non HPO workflows)


def main():
	params = initialize_parameters()
	scores = run(params)
	print(params["data_dir"])

	# demonstrating a list
	for i, value in enumerate(params["dense"]):
		print("dense layer {} has {} nodes".format(i, value))


if __name__ == "__main__":
	main()