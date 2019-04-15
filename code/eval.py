import torch 
import torch.nn
import argparse
import math
import os
from glob import glob

from model import NLIModel
from data import load_SNLI_datasets, debug_level
from mutils import load_model, load_args, args_to_params

from tensorboardX import SummaryWriter



class SNLIEval:

	def __init__(self, model, batch_size=64):
		self.train_dataset, self.val_dataset, self.test_dataset, _, _, _ = load_SNLI_datasets()
		self.model = model
		self.batch_size = batch_size
		self.accuracies = dict()

	def eval(self, iteration=None, dataset=None):
		if dataset is None:
			dataset = self.val_dataset
		self.model.eval()
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / self.batch_size))
		correct_preds = []
		preds_list = []
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
			embeds, lengths, batch_labels = dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True, bidirectional=self.model.is_bidirectional())
			preds = self.model(words_s1 = embeds[0], lengths_s1 = lengths[0], words_s2 = embeds[1], lengths_s2 = lengths[1], applySoftmax=True)
			_, pred_labels = torch.max(preds, dim=-1)
			preds_list += torch.squeeze(pred_labels).tolist()
			correct_preds += torch.squeeze(pred_labels == batch_labels).tolist()
		accuracy = sum(correct_preds) * 1.0 / len(correct_preds)
		print("Evaluation accuracy: %4.2f%%" % (accuracy * 100.0))
		print("Number of predictions: " + ", ".join([("class %i: %i" % (i, sum([j==i for j in preds_list]))) for i in range(3)]))
		if iteration is not None:
			self.accuracies[iteration] = accuracy
		return accuracy

	def test_best_model(self, checkpoint_path, delete_others=False, run_training_set=False):
		final_dict = load_model(checkpoint_path)
		max_acc = max(final_dict["eval_accuracies"])
		best_epoch = final_dict["eval_accuracies"].index(max_acc) + 1
		print("Best epoch: " + str(best_epoch))

		load_model(os.path.join(checkpoint_path, "checkpoint_" + str(best_epoch).zfill(3) + ".tar"), model=self.model)
		if run_training_set:
			train_acc = self.eval(dataset=self.train_dataset)
		val_acc = self.eval(dataset=self.val_dataset)
		test_acc = self.eval(dataset=self.test_dataset)
		if val_acc != max_acc / 100.0:
			print("[!] ERROR: Found different accuracy then reported in the final state dict")
			sys.exit(1)
		if run_training_set:
			print("Train accuracy: %4.2f%%" % (train_acc * 100.0))
		print("Val accuracy: %4.2f%%" % (val_acc * 100.0))
		print("Test accuracy: %4.2f%%" % (test_acc * 100.0))


	def evaluate_all_models(self, checkpoint_path):
		checkpoint_files = sorted(glob(os.path.join(checkpoint_path, "*.tar")))

		model_results = dict()

		for i in range(len(checkpoint_files)):
			checkpoint_dict = load_model(checkpoint_files[i], model=self.model)
			epoch = checkpoint_dict["epoch"]
			model_results[epoch] = dict()
			model_results[epoch]["checkpoint_file"] = checkpoint_files[i]
			model_results[epoch]["train"] = self.eval(dataset=self.train_dataset)
			model_results[epoch]["val"] = self.eval(dataset=self.val_dataset)
			model_results[epoch]["test"] = self.eval(dataset=self.test_dataset)
			print("Model at epoch %i achieved %4.2f%% on validation and %4.2f%% on test dataset" % (epoch, 100.0 * model_results[epoch]["val"], 100.0 * model_results[epoch]["test"]))

		best_acc = {
			"train": {"acc": 0, "epoch": 0},
			"val": {"acc": 0, "epoch": 0},
			"test": {"acc": 0, "epoch": 0}
		}
		for epoch, epoch_dict in model_results.items():
			for data in ["train", "val", "test"]:
				if epoch_dict[data] > best_acc[data]["acc"]:
					best_acc[data]["epoch"] = epoch
					best_acc[data]["acc"] = epoch_dict[data] 

		print("Best train accuracy: %4.2f%% (epoch %i)" % (100.0 * best_acc["train"]["acc"], best_acc["train"]["epoch"]))
		print("Best validation accuracy: %4.2f%% (epoch %i)" % (100.0 * best_acc["val"]["acc"], best_acc["val"]["epoch"]))
		print("Best test accuracy: %4.2f%% (epoch %i)" % (100.0 * best_acc["test"]["acc"], best_acc["test"]["epoch"]))
		return model_results, best_acc


	def visualize_tensorboard(self, checkpoint_path, optimizer_params=None, replace_old_files=True):
		if replace_old_files:
			for old_tf_file in sorted(glob(os.path.join(checkpoint_path, "events.out.tfevents.*"))):
				print("Removing " + old_tf_file + "...")
				os.remove(old_tf_file)
		
		writer = SummaryWriter(log_dir=checkpoint_path)
		
		# dummy_embeds, dummy_length, _ = self.train_dataset.get_batch(self.batch_size, loop_dataset=False, toTorch=True, bidirectional=self.model.is_bidirectional())
		# writer.add_graph(self.model, (dummy_embeds[0], dummy_length[0], dummy_embeds[1], dummy_length[1]))
		
		final_dict = load_model(checkpoint_path)
		for batch in range(len(final_dict["loss_avg_list"])):
			writer.add_scalar("train/loss", final_dict["loss_avg_list"][batch], batch*50+1)

		for epoch in range(len(final_dict["eval_accuracies"])):
			writer.add_scalar("eval/accuracy", final_dict["eval_accuracies"][epoch], epoch+1)

		if optimizer_params is not None:
			lr = optimizer_params["lr"]
			lr_decay_step = optimizer_params["lr_decay_step"]
			for epoch in range(len(final_dict["eval_accuracies"])):
				writer.add_scalar("train/learning_rate", lr, epoch+1)
				if epoch in final_dict["lr_red_step"]:
					lr *= lr_decay_step

		model_results, best_acc = evaluate_all_models(checkpoint_path)
		for epoch, result_dict in model_results.items():
			for data in ["train", "val", "test"]:
				writer.add_scalar("eval/" + data + "_accuracy", result_dict[data], epoch+1)

		writer.close()



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str)
	args = parser.parse_args()

	model_type, model_params, optimizer_params = args_to_params(load_args(args.checkpoint_path))

	_, _, _, _, _, wordvec_tensor = load_SNLI_datasets(debug_dataset = True)
	model = NLIModel(model_type, model_params, wordvec_tensor)

	evaluater = SNLIEval(model)
	# evaluater.evaluate_all_models(args.checkpoint_path)
	# evaluater.test_best_model(args.checkpoint_path)
	evaluater.visualize_tensorboard(args.checkpoint_path, optimizer_params=optimizer_params)