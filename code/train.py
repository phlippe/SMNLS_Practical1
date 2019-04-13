import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
import datetime
import os
import sys
import json
import pickle
from glob import glob

from eval import SNLIEval
from model import NLIModel
from data import load_SNLI_datasets

PARAM_CONFIG_FILE = "param_config.pik"

class SNLITrain:

	OPTIMIZER_SGD = 0
	OPTIMIZER_ADAM = 1


	def __init__(self, model_type, model_params, optimizer_params, batch_size, checkpoint_path):
		self.train_dataset, _, _, _, self.word2id, wordvec_tensor = load_SNLI_datasets(debug_dataset = False)
		self.model = NLIModel(model_type, model_params, wordvec_tensor)
		self.evaluater = SNLIEval(self.model)
		self.batch_size = batch_size
		self._create_optimizer(optimizer_params)
		self._prepare_checkpoint(checkpoint_path)
		

	def _create_optimizer(self, optimizer_params):
		if optimizer_params["optimizer"] == SNLITrain.OPTIMIZER_SGD:
			self.optimizer = torch.optim.SGD(self.model.parameters(), 
											 lr=optimizer_params["lr"], 
											 weight_decay=optimizer_params["weight_decay"])
		elif optimizer_params["optimizer"] == SNLITrain.OPTIMIZER_ADAM:
			self.optimizer = torch.optim.Adam(self.model.parameters(), 
											  lr=optimizer_params["lr"])
		else:
			print("[!] ERROR: Unknown optimizer: " + str(optimizer_params["optimizer"]))
			sys.exit(1)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=optimizer_params["lr_decay_step"])
		self.max_red_steps = optimizer_params["lr_max_red_steps"]


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def _get_dict_val(self, checkpoint_dict, key, default_val):
		if key in checkpoint_dict:
			return checkpoint_dict[key]
		else:
			return default_val


	def train_model(self, epochs=50, loss_freq=50):

		loss_module = nn.CrossEntropyLoss()
		if torch.cuda.is_available():
			loss_module = loss_module.cuda()

		checkpoint_dict = self.load_model()
		start_epoch = self._get_dict_val(checkpoint_dict, "epoch", 0)
		eval_accuracies = self._get_dict_val(checkpoint_dict, "eval_accuracies", list())
		loss_avg_list = self._get_dict_val(checkpoint_dict, "loss_avg_list", list())
		lr_red_step = self._get_dict_val(checkpoint_dict, "lr_red_step", list())
		
		for index_epoch in range(start_epoch, epochs):

			self.model.train()
			num_steps = int(math.ceil(self.train_dataset.get_num_examples() * 1.0 / self.batch_size))
			loss_avg_list.append(0)
			for step_index in range(num_steps):

				embeds, lengths, batch_labels = self.train_dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True, bidirectional=self.model.is_bidirectional())
				preds = self.model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=False)
				loss = loss_module(preds, batch_labels)

				self.model.zero_grad()
				loss.backward()
				self.optimizer.step()

				loss_avg_list[-1] += loss.item()
				if (step_index + 1) % loss_freq == 0:
					loss_avg_list[-1] = loss_avg_list[-1]  / loss_freq
					print("Training epoch %i|%i, step %i|%i. Loss: %6.5f" % (index_epoch+1, epochs, step_index+1, num_steps, loss_avg_list[-1]))
					loss_avg_list.append(0)
			del loss_avg_list[-1]

			acc = self.evaluater.eval(index_epoch)
			eval_accuracies.append(acc)

			if len(eval_accuracies) > 2:
				if eval_accuracies[-1] < (eval_accuracies[-2] + eval_accuracies[-3]) / 2:
					print("Reducing learning rate")
					self.lr_scheduler.step()
					lr_red_step.append(index_epoch + 1)

			checkpoint_dict = {
				"epoch": index_epoch + 1,
				"eval_accuracies": eval_accuracies,
				"loss_avg_list": loss_avg_list,
				"lr_red_step": lr_red_step
			}
			self.save_model(index_epoch + 1, checkpoint_dict)

			if len(lr_red_step) > self.max_red_steps:
				print("Reached maximum number of learning rate reduction steps")
				break

		with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
			f.write("".join(["Epoch %i: %4.2f%%\n" % (i+1,eval_accuracies[i]*100.0) for i in range(len(eval_accuracies))]))
			f.write("Best accuracy achieved: %4.2f%%" % (max(eval_accuracies) * 100.0))


	def load_model(self):
		checkpoint_files = sorted(glob(os.path.join(self.checkpoint_path, "*.tar")))
		if len(checkpoint_files) == 0:
			return 0, list()
		latest_checkpoint = checkpoint_files[-1]
		print("Loading checkpoint \"" + str(latest_checkpoint) + "\"")
		checkpoint = torch.load(latest_checkpoint)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		add_param_dict = dict()
		for key, val in checkpoint.items():
			if "state_dict" not in key:
				add_param_dict[key] = val
		return add_param_dict


	def save_model(self, epoch, add_param_dict):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(epoch).zfill(3) + ".tar")
		checkpoint_dict = {
				'model_state_dict': self.model.state_dict(),
				'optimizer_state_dict': self.optimizer.state_dict(),
				'scheduler_state_dict': self.lr_scheduler.state_dict()
		}
		for key, val in add_param_dict.items():
			checkpoint_dict[key] = val
		torch.save(checkpoint_dict, checkpoint_file)







if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--epochs", help="Maximum number of epochs to train. Default: dynamic with learning rate threshold", type=int, default=-1)
	parser.add_argument("--eval_freq", help="Frequency of evaluation on validation set (in number of steps/iterations). Default: once per epoch", type=int, default=-1)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=0.1)
	parser.add_argument("--lr_decay", help="Decay of learning rate of the optimizer. Always applied if eval accuracy droped compared to mean of last two epochs", type=float, default=0.2)
	parser.add_argument("--lr_max_red_steps", help="Maximum number of times learning rate should be decreased before terminating", type=int, default=4)
	parser.add_argument("--weight_decay", help="Weight decay of the SGD optimizer", type=float, default=1e-2)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam", type=int, default=0)
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	parser.add_argument("--fc_dim", help="Number of hidden units in fully connected layers (classifier)", type=int, default=512)
	parser.add_argument("--fc_dropout", help="Dropout probability in FC classifier", type=float, default=0.0)
	parser.add_argument("--embed_dim", help="Embedding dimensionality of sentence", type=int, default=2048)
	parser.add_argument("--model", help="Which encoder model to use. 0: BOW, 1: LSTM, 2: Bi-LSTM, 3: Bi-LSTM with max pooling", type=int, default=0)
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)

	args = parser.parse_args()
	print(args)

	if args.load_config:
		if args.checkpoint_path is None:
			print("[!] ERROR: Please specify the checkpoint path to load the config from.")
			sys.exit(1)
		param_file_path = os.path.join(args.checkpoint_path, PARAM_CONFIG_FILE)
		with open(param_file_path, "rb") as f:
			print("Loading parameter configuration from \"" + str(args.checkpoint_path) + "\"")
			args = pickle.load(f)

	# Define model parameters
	model_params = {
		"embed_word_dim": 300,
		"embed_sent_dim": args.embed_dim,
		"fc_dropout": args.fc_dropout, 
		"fc_dim": args.fc_dim,
		"n_classes": 3
	}
	if args.model == NLIModel.AVERAGE_WORD_VECS:
		model_params["embed_sent_dim"] = 300

	optimizer_params = {
		"optimizer": args.optimizer,
		"lr": args.learning_rate,
		"weight_decay": args.weight_decay,
		"lr_decay_step": args.lr_decay,
		"lr_max_red_steps": args.lr_max_red_steps
	}

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available: 
		torch.cuda.manual_seed_all(args.seed)

	# Setup training
	trainModule = SNLITrain(model_type=args.model, 
							model_params=model_params,
							optimizer_params=optimizer_params, 
							batch_size=args.batch_size,
							checkpoint_path=args.checkpoint_path
							)

	with open(os.path.join(trainModule.checkpoint_path, PARAM_CONFIG_FILE), "wb") as f:
		pickle.dump(args, f)

	trainModule.train_model(50, loss_freq=50)
