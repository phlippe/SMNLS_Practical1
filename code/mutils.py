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
	checkpoint = torch.load(checkpoint_path)
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
		"lr_max_red_steps": args.lr_max_red_steps
	}

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available: 
		torch.cuda.manual_seed_all(args.seed)

	return args.model, model_params, optimizer_params

def get_dict_val(checkpoint_dict, key, default_val):
	if key in checkpoint_dict:
		return checkpoint_dict[key]
	else:
		return default_val
