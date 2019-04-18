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
from shutil import copyfile

from model import NLIModel
from data import load_SNLI_datasets, debug_level, set_debug_level

PARAM_CONFIG_FILE = "param_config.pik"


def load_model(checkpoint_path, model=None, optimizer=None, lr_scheduler=None):
	if os.path.isdir(checkpoint_path):
		checkpoint_files = sorted(glob(os.path.join(checkpoint_path, "*.tar")))
		if len(checkpoint_files) == 0:
			return dict()
		checkpoint_path = checkpoint_files[-1]
	print("Loading checkpoint \"" + str(checkpoint_path) + "\"")
	if torch.cuda.is_available():
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
	if model is not None:
		model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer is not None:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if lr_scheduler is not None:
		lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	add_param_dict = dict()
	for key, val in checkpoint.items():
		if "state_dict" not in key:
			add_param_dict[key] = val
	return add_param_dict

def load_model_from_args(args, checkpoint_path=None):
	model_type, model_params, optimizer_params = args_to_params(args)
	_, _, _, _, _, wordvec_tensor = load_SNLI_datasets(debug_dataset = False)
	model = NLIModel(model_type, model_params, wordvec_tensor)
	if checkpoint_path is not None:
		load_model(checkpoint_path, model=model)
	return model

def load_args(checkpoint_path):
	if os.path.isfile(checkpoint_path):
		checkpoint_path = checkpoint_path.rsplit("/",1)[0]
	param_file_path = os.path.join(checkpoint_path, PARAM_CONFIG_FILE)
	if not os.path.exists(param_file_path):
		print("[!] ERROR: Could not find parameter config file: " + str(param_file_path))
	with open(param_file_path, "rb") as f:
		print("Loading parameter configuration from \"" + str(param_file_path) + "\"")
		args = pickle.load(f)
	return args


def args_to_params(args):
	# Define model parameters
	model_params = {
		"embed_word_dim": 300,
		"embed_sent_dim": args.embed_dim,
		"fc_dropout": args.fc_dropout, 
		"fc_dim": args.fc_dim,
		"fc_nonlinear": args.fc_nonlinear,
		"n_classes": 3
	}
	if args.model == NLIModel.AVERAGE_WORD_VECS:
		model_params["embed_sent_dim"] = 300

	optimizer_params = {
		"optimizer": args.optimizer,
		"lr": args.learning_rate,
		"weight_decay": args.weight_decay,
		"lr_decay_step": args.lr_decay,
		"lr_max_red_steps": args.lr_max_red_steps,
		"momentum": args.momentum if hasattr(args, "momentum") else 0.0
	}

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available: 
		torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False	

	return args.model, model_params, optimizer_params

def get_dict_val(checkpoint_dict, key, default_val):
	if key in checkpoint_dict:
		return checkpoint_dict[key]
	else:
		return default_val

def visualize_tSNE(model, dataset, tensorboard_writer, batch_size=64, embedding_name='default', global_step=None):

	number_batches = int(math.ceil(dataset.get_num_examples()/batch_size))
	data_embed_list = list()
	label_list = list()
	for i in range(number_batches):
		print("Processed %4.2f%% of dataset" % (i*100.0/number_batches), end="\r")
		embeds, lengths, labels = dataset.get_batch(batch_size=batch_size, loop_dataset=False, toTorch=True)
		embeds = embeds[0] if isinstance(embeds, list) else embeds 
		lengths = lengths[0] if isinstance(lengths, list) else lengths
		sent_embeds = model.encode_sentence(embeds, lengths)
		data_embed_list.append(sent_embeds)
		label_list.append(labels)
	final_embeddings = torch.cat(data_embed_list, dim=0)
	final_labels = torch.cat(label_list, dim=0)
	tensorboard_writer.add_embedding(final_embeddings, metadata=final_labels, tag=embedding_name, global_step=global_step)

def copy_results():
	checkpoint_folder = sorted(glob("checkpoints/*"))
	for check_dir in checkpoint_folder:
		if os.path.isfile(os.path.join(check_dir, "evaluation.txt")):
			result_folder = os.path.join("results/", check_dir.split("/")[-1])
			if not os.path.exists(result_folder):
				os.makedirs(result_folder)
			for file_to_copy in ["sent_eval.pik", "results.txt", "param_config.pik", "evaluation.txt", "extra_evaluation.txt"]:
				copyfile(src=os.path.join(check_dir, file_to_copy), 
						 dst=os.path.join(result_folder, file_to_copy.split("/")[-1]))
			for tf_file_to_copy in sorted(glob(os.path.join(check_dir, "*tfevents*"))):
				copyfile(src=tf_file_to_copy,
						 dst=os.path.join(result_folder, tf_file_to_copy.split("/")[-1]))

			result_file = os.path.join(check_dir, "results.txt")
			if os.path.isfile(result_file):
				with open(result_file, "r") as f:
					eval_lines = f.readlines()
				max_acc = -1
				max_epoch = -1
				for line in eval_lines:
					if line.split(" ")[0] != "Epoch":
						continue
					loc_epoch = int(line.split(" ")[1].replace(":",""))
					loc_acc = float(line.split(" ")[2].replace("%",""))
					if loc_acc > max_acc:
						max_acc = loc_acc
						max_epoch = loc_epoch
				checkpoint_file = "checkpoint_" + str(max_epoch).zfill(3) + ".tar"
				if max_epoch >= 1 and not os.path.exists(os.path.join(result_folder, checkpoint_file)):
					copyfile(src=os.path.join(check_dir, checkpoint_file),
							 dst=os.path.join(result_folder, checkpoint_file))

if __name__ == '__main__':
	copy_results()