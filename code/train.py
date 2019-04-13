import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
from eval import SNLIEval
from model import NLIModel
from data import load_SNLI_datasets


class SNLITrain:

	def __init__(self, model_type, model_params, batch_size, learning_rate, weight_decay):
		self.train_dataset, _, _, _, self.word2id, wordvec_tensor = load_SNLI_datasets(debug_dataset = True)
		self.model = NLIModel(model_type, model_params, wordvec_tensor)
		self.evaluater = SNLIEval(self.model)
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay

	def train_model(self, epochs, loss_freq=20):

		optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

		loss_module = nn.CrossEntropyLoss()
		if torch.cuda.is_available():
			loss_module = loss_module.cuda()

		loss_avg_list = [0]
		eval_accuracies = []

		for index_epoch in range(epochs):

			self.model.train()
			num_steps = int(math.ceil(self.train_dataset.get_num_examples() * 1.0 / self.batch_size))
			for step_index in range(num_steps):

				embeds, lengths, batch_labels = self.train_dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True)
				preds = self.model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=True)
				loss = loss_module(preds, batch_labels)

				self.model.zero_grad()
				loss.backward()
				optimizer.step()

				loss_avg_list[-1] += loss.item() / loss_freq
				if (step_index + 1) % loss_freq == 0:
					print("Training epoch %i|%i, step %i|%i. Loss: %4.2f" % (index_epoch+1, epochs, step_index+1, num_steps, loss_avg_list[-1]))
					loss_avg_list.append(0)

			acc = self.evaluater.eval(index_epoch)
			eval_accuracies.append(acc)





if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--epochs", help="Number of epochs to train. Default: dynamic with learning rate threshold", type=int, default=-1)
	parser.add_argument("--eval_freq", help="Frequency of evaluation on validation set (in number of steps/iterations). Default: once per epoch", type=int, default=-1)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=0.1)
	parser.add_argument("--weight_decay", help="Weight decay of the SGD optimizer", type=float, default=1e-2)
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default="")
	parser.add_argument("--fc_dim", help="Number of hidden units in fully connected layers (classifier)", type=int, default=512)
	parser.add_argument("--fc_dropout", help="Dropout probability in FC classifier", type=float, default=0.0)
	parser.add_argument("--embed_dim", help="Embedding dimensionality of sentence", type=int, default=2048)
	parser.add_argument("--model", help="Which encoder model to use. 0: BOW, 1: LSTM, 2: Bi-LSTM, 3: Bi-LSTM with max pooling", type=int, default=0)
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)

	args = parser.parse_args()

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

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available: 
		torch.cuda.manual_seed_all(args.seed)

	# Setup training
	trainModule = SNLITrain(model_type=args.model, 
							model_params=model_params, 
							batch_size=args.batch_size,
							learning_rate=args.learning_rate,
							weight_decay=args.weight_decay
							)
	trainModule.train_model(5, loss_freq=200)
