import torch 
import torch.nn
import argparse
import math
from data import load_SNLI_datasets


class SNLIEval:

	def __init__(self, model, validation_set=True, batch_size=64):
		if validation_set:
			_, self.eval_dataset, _, _, _, _ = load_SNLI_datasets()
		else:
			_, _, self.eval_dataset, _, _, _ = load_SNLI_datasets()
		self.model = model
		self.batch_size = batch_size
		self.accuracies = dict()

	def eval(self, iteration=None):
		self.model.eval()
		number_batches = int(math.ceil(self.eval_dataset.get_num_examples() * 1.0 / self.batch_size))
		correct_preds = []
		for batch_ind in range(number_batches):
			print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
			embeds, lengths, batch_labels = self.eval_dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True)
			preds = self.model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=True)
			_, pred_labels = torch.max(preds, dim=-1)
			correct_preds += torch.squeeze(pred_labels == batch_labels).tolist()
		accuracy = sum(correct_preds) * 1.0 / len(correct_preds)
		print("Evaluation accuracy: %4.2f%%" % (accuracy * 100.0))
		if iteration is not None:
			self.accuracies[iteration] = accuracy
		return accuracy
