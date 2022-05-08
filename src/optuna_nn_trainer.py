import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

import optuna
from optuna.trial import TrialState
from optuna.samplers import GridSampler

import util
from nn_trainer import *
from training_data_wrapper import *
from drugcell_nn import *


class OptunaNNTrainer(NNTrainer):

	def __init__(self, data_wrapper):
		super().__init__(data_wrapper)
		

	def exec_study(self):
		search_space = {
			"genotype_hiddens": [2, 4],
			"lr": [1e-4, 1.2e-4, 1.5e-4, 1.8e-4, 2e-4, 3e-4, 4e-4, 5e-4]
		}
		study = optuna.create_study(sampler=GridSampler(search_space), direction="maximize")
		study.optimize(self.train_model, n_trials=15)
		return self.print_result(study)


	def setup_trials(self, trial):

		self.data_wrapper.genotype_hiddens = trial.suggest_categorical("genotype_hiddens", [2, 4])
		self.data_wrapper.lr = trial.suggest_float("lr", 1e-4, 6e-4, log=True)

		batch_size = self.data_wrapper.batchsize
		if batch_size > len(self.train_feature)/4:
			batch_size = 2 ** int(math.log(len(self.train_feature)/4, 2))
			self.data_wrapper.batchsize = trial.suggest_categorical("batchsize", [batch_size])

		for key, value in trial.params.items():
			print("{}: {}".format(key, value))


	def train_model(self, trial):

		epoch_start_time = time.time()
		max_corr = 0.0
		min_loss = None
		early_stopping_counter = 0
		train_corr_at_min_loss = 0.0

		self.setup_trials(trial)

		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=True)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		optimizer.zero_grad()

		print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\telapsed_time")
		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(train_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				# Forward + Backward + Optimize
				optimizer.zero_grad()  # zero the gradient buffer

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final':
						total_loss += loss(output, cuda_labels)
					else:
						total_loss += self.data_wrapper.alpha * loss(output, cuda_labels)
				total_loss.backward()

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				optimizer.step()

			train_corr = util.pearson_corr(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)

			val_loss = 0
			for i, (inputdata, labels) in enumerate(val_loader):
				# Convert torch tensor to Variable
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				aux_out_map, _ = self.model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
					val_label_gpu = cuda_labels
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
					val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

				for name, output in aux_out_map.items():
					loss = nn.MSELoss()
					if name == 'final':
						val_loss += loss(output, cuda_labels)

			val_corr = util.pearson_corr(val_predict, val_label_gpu)

			epoch_end_time = time.time()
			true_auc = torch.mean(train_label_gpu)
			pred_auc = torch.mean(train_predict)
			print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, train_corr, total_loss, true_auc, pred_auc, val_corr, val_loss, epoch_end_time - epoch_start_time))
			epoch_start_time = epoch_end_time

			trial.report(val_corr, epoch)

			if min_loss == None:
				min_loss = val_loss
			elif min_loss - val_loss > self.data_wrapper.delta:
				min_loss = val_loss
				early_stopping_counter = 0
				max_corr = val_corr
				train_corr_at_min_loss = train_corr
			elif min_loss - val_loss < self.data_wrapper.delta:
				early_stopping_counter += 1
				if early_stopping_counter >= self.data_wrapper.patience:
					break

		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

		#torch.save(self.model, self.data_wrapper.modeldir + '/model_trial_' + str(trial.number) + '.pt')
		return max_corr


	def print_result(self, study):

		pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
		complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

		print("Study statistics:")
		print("Number of finished trials:", len(study.trials))
		print("Number of pruned trials:", len(pruned_trials))
		print("Number of complete trials:", len(complete_trials))

		print("Best trial:")
		best_trial = study.best_trial

		print("Value: ", best_trial.value)

		best_params = {}
		print("Params:")
		for key, value in best_trial.params.items():
			print("{}: {}".format(key, value))
			best_params[key] = value
		for key, value in best_trial.user_attrs.items():
			print("{}: {}".format(key, value))
			best_params[key] = value

		return best_params
