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
import copy
from glob import glob
from shutil import copyfile

from model import NLIModel
from data import load_SNLI_datasets, debug_level, set_debug_level, DatasetTemplate, SentData

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
		pretrained_model_dict = {key: val for key, val in checkpoint['model_state_dict'].items() if not key.startswith("embeddings")}
		model_dict = model.state_dict()
		model_dict.update(pretrained_model_dict)
		model.load_state_dict(model_dict)
		# model.load_state_dict(checkpoint['model_state_dict'])
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

def visualize_tSNE(model, dataset, tensorboard_writer, batch_size=64, embedding_name='default', global_step=None, add_reduced_version=False):
	if add_reduced_version:
		random.seed(42)
		sub_dataset = copy.deepcopy(dataset)
		random.shuffle(sub_dataset.data_list)
		sub_datalist = sub_dataset.data_list[:min(1000, len(sub_dataset.data_list))]
		sub_dataset.set_data_list(sub_datalist)
		return visualize_tSNE(model=model, dataset=sub_dataset, tensorboard_writer=tensorboard_writer, 
							  batch_size=batch_size, embedding_name=embedding_name + "_reduced", 
					   		  global_step=global_step, add_reduced_version=False)
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
	final_labels = torch.cat(label_list, dim=0).squeeze().tolist()
	final_labels = [dataset.label_to_string(lab) for lab in final_labels]
	tensorboard_writer.add_embedding(final_embeddings, metadata=final_labels, tag=embedding_name, global_step=global_step)


# Function copied from SentEval for reproducibility
def loadFile(fpath):
	with open(fpath, 'r', encoding='latin-1') as f:
		return [line.split() for line in f.read().splitlines()]

def task_to_dataset(sentences, labels, label_dict=None):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = False)
	data_batch = list()
	for sent, lab in zip(sentences, labels):
		str_sent = " ".join([w if isinstance(w, str) else w.decode('UTF-8') for w in sent])
		new_d = SentData(sentence=str_sent, label=lab)
		new_d.translate_to_dict(word2id)
		data_batch.append(new_d)
	dataset = DatasetTemplate("all")
	dataset.set_data_list(data_batch)
	if label_dict is not None:
		dataset.add_label_explanation(label_dict)
	return dataset

def get_transfer_datasets():
	transfer_datasets = dict()
	
	def load_classification_dataset(file_classes, label_dict=None):
		sents_classes = [loadFile(fpath) for fpath in file_classes]
		sentences = []
		labels = []
		for class_index, sent_class in enumerate(sents_classes):
			sentences += sent_class
			labels += [class_index] * len(sent_class)
		return task_to_dataset(sentences, labels, label_dict=label_dict)

	# SUBJ Task
	subj_task_path = "../../SentEval/data/downstream/SUBJ/"		
	transfer_datasets["SUBJ"] = load_classification_dataset([os.path.join(subj_task_path, 'subj.objective'),
															 os.path.join(subj_task_path, 'subj.subjective')],
															{0: "Objective", 1: "Subjective"})

	cr_task_path = "../../SentEval/data/downstream/CR/"		
	transfer_datasets["CR"] = load_classification_dataset([os.path.join(cr_task_path, 'custrev.neg'),
														   os.path.join(cr_task_path, 'custrev.pos')],
														   {0: "Negative", 1: "Positive"})

	mr_task_path = "../../SentEval/data/downstream/MR/"		
	transfer_datasets["MR"] = load_classification_dataset([os.path.join(mr_task_path, 'rt-polarity.neg'),
														   os.path.join(mr_task_path, 'rt-polarity.pos')],
														   {0: "Negative", 1: "Positive"})

	mpqa_task_path = "../../SentEval/data/downstream/MPQA/"		
	transfer_datasets["MPQA"] = load_classification_dataset([os.path.join(mpqa_task_path, 'mpqa.neg'),
															 os.path.join(mpqa_task_path, 'mpqa.pos')],
														     {0: "Negative", 1: "Positive"})

	trec_task_path = "../../SentEval/data/downstream/TREC/"		
	trec_file = loadFile(os.path.join(trec_task_path, 'train_5500.label'))
	tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
			   'HUM': 3, 'LOC': 4, 'NUM': 5}
	trec_sentences = [line[1:] for line in trec_file]
	trec_short_sentences = [line[1:min(3, len(line))] for line in trec_file]
	trec_labels = [tgt2idx[line[0].split(":")[0]] for line in trec_file]
	transfer_datasets["TREC"] = task_to_dataset(trec_sentences, trec_labels, label_dict=tgt2idx)
	transfer_datasets["TREC_short"] = task_to_dataset(trec_short_sentences, trec_labels, label_dict=tgt2idx)

	return transfer_datasets



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
				last_epoch = -1
				for line in eval_lines:
					if line.split(" ")[0] != "Epoch":
						continue
					loc_epoch = int(line.split(" ")[1].replace(":",""))
					loc_acc = float(line.split(" ")[2].replace("%",""))
					if loc_acc > max_acc:
						max_acc = loc_acc
						max_epoch = loc_epoch
					if loc_epoch > last_epoch:
						last_epoch = loc_epoch
				for epoch in [max_epoch, last_epoch]:
					checkpoint_file = "checkpoint_" + str(epoch).zfill(3) + ".tar"
					if epoch >= 1 and not os.path.exists(os.path.join(result_folder, checkpoint_file)):
						copyfile(src=os.path.join(check_dir, checkpoint_file),
								 dst=os.path.join(result_folder, checkpoint_file))

def results_to_table():
	result_folder = sorted(glob("results/*"))
	s = "| Experiment names | Train | Val | Test | Test easy | Test hard | Micro | Macro |\n"
	s += "| " + " | ".join(["---"]*(len(s.split("|"))-2)) + " |\n"
	for res_dir in result_folder:
		print("Processing " + str(res_dir))
		s += "| " + res_dir.split("/")[-1] + " | "
		with open(os.path.join(res_dir, "evaluation.txt"), "r") as f:
			lines = f.readlines()
		if len(lines) > 3:
			for i in [1, 2, 3]:
				s += lines[i].split(" ")[-1].replace("\n","") + " | "
		else:
			s += " - | "
			for i in [1, 2]:
				s += lines[i].split(" ")[-1].replace("\n","") + " | "
		with open(os.path.join(res_dir, "extra_evaluation.txt"), "r") as f:
			for line in f.readlines():
				s += line.split(" ")[-1].replace("\n","") + " | "
		with open(os.path.join(res_dir, "sent_eval.pik"), "rb") as f:
			sent_eval_dict = pickle.load(f)
		accs = [val["acc"] for key, val in sent_eval_dict.items() if "acc" in val]
		s += "%4.2f%% | %4.2f%% |" % (sum(accs)/len(accs), sum(accs)/len(accs))
		s += "\n"
	print(s)

def sent_eval_to_table():
	result_folder = sorted(glob("results/*"))
	with open(os.path.join(result_folder[0], "sent_eval.pik"), "rb") as f:
		sample_dict = pickle.load(f)
	task_list = list(sample_dict.keys())
	s = "| Experiment names | " + " | ".join(task_list) + " |\n"
	s += "| " + " | ".join(["---"]*(len(task_list)+1)) + " |\n"
	for res_dir in result_folder:
		s += "| " + res_dir.split("/")[-1] + " | "
		with open(os.path.join(res_dir, "sent_eval.pik"), "rb") as f:
			sample_dict = pickle.load(f)
		for task_key in task_list:
			if "acc" in sample_dict[task_key]:
				s +=  "%4.2f%%" % (sample_dict[task_key]["acc"]) + ("/%4.2f%%" % (sample_dict[task_key]["f1"]) if "f1" in sample_dict[task_key] else "")
			elif "pearson" in sample_dict[task_key]:
				s += "%4.2f" % (sample_dict[task_key]["pearson"])
			elif 'all' in sample_dict[task_key]:
				s += "%4.2f/%4.2f" % (sample_dict[task_key]["all"]["pearson"]["wmean"], sample_dict[task_key]["all"]["spearman"]["wmean"])
			s +=  " | "
		s += "\n"
	print(s)




if __name__ == '__main__':
	copy_results()
	# results_to_table()
	# print("\n\n")
	# sent_eval_to_table()

