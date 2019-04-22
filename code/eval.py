import torch 
import torch.nn
import argparse
import math
import os
import sys
import json
import pickle
import numpy as np
from glob import glob

from model import NLIModel
from data import load_SNLI_datasets, load_SNLI_splitted_test, debug_level
from mutils import load_model, load_model_from_args, load_args, args_to_params, visualize_tSNE, get_transfer_datasets

from tensorboardX import SummaryWriter

from sent_eval import perform_SentEval



class SNLIEval:

	def __init__(self, model, batch_size=64):
		self.train_dataset, self.val_dataset, self.test_dataset, _, _, _ = load_SNLI_datasets(debug_dataset = False)
		self.test_hard_dataset, self.test_easy_dataset = load_SNLI_splitted_test()
		self.model = model
		self.batch_size = batch_size
		self.accuracies = dict()

	def eval(self, iteration=None, dataset=None, ret_pred_list=False):
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
		if not ret_pred_list:
			return accuracy
		else:
			return accuracy, preds_list

	def test_best_model(self, checkpoint_path, delete_others=False, run_standard_eval=True, run_training_set=False, run_sent_eval=True, run_extra_eval=True, light_senteval=True):
		final_dict = load_model(checkpoint_path)
		max_acc = max(final_dict["eval_accuracies"])
		best_epoch = final_dict["eval_accuracies"].index(max_acc) + 1
		s = "Best epoch: " + str(best_epoch) + " with accuracy %4.2f%%" % (max_acc * 100.0) + "\n"
		print(s)

		best_checkpoint_path = os.path.join(checkpoint_path, "checkpoint_" + str(best_epoch).zfill(3) + ".tar")
		load_model(best_checkpoint_path, model=self.model)
		for param in self.model.parameters():
			param.requires_grad = False

		if run_standard_eval:
			if run_training_set:
				# For training, we evaluate on the very last checkpoint as we expect to have the best training performance there
				load_model(checkpoint_path, model=self.model)
				train_acc = self.eval(dataset=self.train_dataset)
				# Load best checkpoint again
				load_model(best_checkpoint_path, model=self.model)
			val_acc = self.eval(dataset=self.val_dataset)
			test_acc, pred_list = self.eval(dataset=self.test_dataset, ret_pred_list=True)
			if abs(val_acc - max_acc) > 0.0005:
				print("[!] ERROR: Found different accuracy then reported in the final state dict. Difference: %f" % (100.0 * abs(val_acc - max_acc)) ) 
				return 
			np.save(os.path.join(checkpoint_path, "test_predictions.npy"), np.array(pred_list))
			if run_training_set:
				s += ("Train accuracy: %4.2f%%" % (train_acc * 100.0)) + "\n"
			s += ("Val accuracy: %4.2f%%" % (val_acc * 100.0)) + "\n"
			s += ("Test accuracy: %4.2f%%" % (test_acc * 100.0)) + "\n"

			with open(os.path.join(checkpoint_path, "evaluation.txt"), "w") as f:
				f.write(s)

		if run_extra_eval:
			test_easy_acc = self.eval(dataset=self.test_easy_dataset)
			test_hard_acc = self.eval(dataset=self.test_hard_dataset)
			s = "Test easy accuracy: %4.2f%%\n Test hard accuracy: %4.2f%%\n" % (test_easy_acc*100.0, test_hard_acc*100.0)
			with open(os.path.join(checkpoint_path, "extra_evaluation.txt"), "w") as f:
				f.write(s)

		if run_sent_eval:
			self.model.eval()
			res = perform_SentEval(self.model, fast_eval=light_senteval)
			with open(os.path.join(checkpoint_path, "sent_eval.pik"), "wb") as f:
				pickle.dump(res, f)


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


	def visualize_tensorboard(self, checkpoint_path, optimizer_params=None, replace_old_files=False, additional_datasets=None):
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

		# model_results, best_acc = self.evaluate_all_models(checkpoint_path)
		# for epoch, result_dict in model_results.items():
		# 	for data in ["train", "val", "test"]:
		# 		writer.add_scalar("eval/" + data + "_accuracy", result_dict[data], epoch+1)

		max_acc = max(final_dict["eval_accuracies"])
		best_epoch = final_dict["eval_accuracies"].index(max_acc) + 1
		load_model(os.path.join(checkpoint_path, "checkpoint_" + str(best_epoch).zfill(3) + ".tar"), model=self.model)

		visualize_tSNE(self.model, self.test_easy_dataset, writer, embedding_name="Test set easy", add_reduced_version=True)
		visualize_tSNE(self.model, self.test_hard_dataset, writer, embedding_name="Test set hard", add_reduced_version=True)
		if additional_datasets is not None:
			for dataset_name, dataset in additional_datasets.items():
				print("Adding embeddings for dataset " + str(dataset_name))
				visualize_tSNE(self.model, dataset, writer, embedding_name=dataset_name, add_reduced_version=True)

		writer.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints are saved. If it is a regex expression, all experiments are evaluated", type=str, required=True)
	parser.add_argument("--overwrite", help="Whether evaluations should be re-run if there already exists an evaluation file.", action="store_true")
	parser.add_argument("--visualize_embeddings", help="Whether the embeddings of the model should be visualized or not", action="store_true")
	parser.add_argument("--full_senteval", help="Whether to run SentEval with the heavy setting or not", action="store_true")
	# parser.add_argument("--all", help="Evaluating all experiments in the checkpoint folder (specified by checkpoint path) if not already done", action="store_true")
	args = parser.parse_args()
	model_list = sorted(glob(args.checkpoint_path))
	transfer_datasets = get_transfer_datasets()

	for model_checkpoint in model_list:
		if not os.path.isfile(os.path.join(model_checkpoint, "results.txt")):
			print("Skipped " + str(model_checkpoint) + " because of missing results file." )
			continue
		
		skip_standard_eval = not args.overwrite and os.path.isfile(os.path.join(model_checkpoint, "evaluation.txt"))
		skip_sent_eval = not args.overwrite and os.path.isfile(os.path.join(model_checkpoint, "sent_eval.pik"))
		skip_extra_eval = (not args.overwrite and os.path.isfile(os.path.join(model_checkpoint, "extra_evaluation.txt")))

		try:
			model = load_model_from_args(load_args(model_checkpoint))
			evaluater = SNLIEval(model)
			evaluater.test_best_model(model_checkpoint, 
									  run_standard_eval=(not skip_standard_eval), 
									  run_training_set=True,
									  run_sent_eval=(not skip_sent_eval),
									  run_extra_eval=(not skip_extra_eval),
									  light_senteval=(not args.full_senteval))
			if args.visualize_embeddings:
				evaluater.visualize_tensorboard(model_checkpoint, replace_old_files=args.overwrite, additional_datasets=transfer_datasets)
		except RuntimeError as e:
			print("[!] Runtime error while loading " + model_checkpoint)
			print(e)
			continue
	# evaluater.evaluate_all_models(args.checkpoint_path)
	# evaluater.visualize_tensorboard(args.checkpoint_path, optimizer_params=optimizer_params)