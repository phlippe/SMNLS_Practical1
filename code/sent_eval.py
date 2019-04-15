import torch 
import torch.nn
import argparse
import math
import os
import sys
import logging
from glob import glob

from model import NLIModel
from data import load_SNLI_datasets, debug_level, NLIData, SNLIDataset
from mutils import load_model, load_args, args_to_params

# Sent initials for SentEval
PATH_TO_SENTEVAL = "../../SentEval"
PATH_TO_DATA = PATH_TO_SENTEVAL + '/data'
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def create_model(checkpoint_path, model_type, model_params):
	_, _, _, word2vec, word2id, wordvec_tensor = load_SNLI_datasets(debug_dataset = True)
	model = NLIModel(model_type, model_params, wordvec_tensor)
	_ = load_model(checkpoint_path, model=model)
	for param in model.parameters():
		param.requires_grad = False
	return model

# SentEval prepare and batcher
def prepare(params, samples):
	_, _, _, _, word2id, _ = load_SNLI_datasets(debug_dataset = True)
	params.word2id = word2id
	words = ' '.join([' '.join(s).lower() for s in samples]).split(" ")
	unknown_words = dict()
	for w in words:
		if w not in word2id:
			unknown_words[w] = ''
	print("Number of unknown words: " + str(len(unknown_words.keys())))
	return

def batcher(params, batch):
	global MODEL
	data_batch = list()
	for sent in batch:
		new_d = NLIData(premise=" ".join(sent), hypothesis='.', label='-')
		new_d.translate_to_dict(params.word2id)
		data_batch.append(new_d)
	sents, lengths, _ = SNLIDataset.sents_to_Tensors([[d.premise_vocab for d in data_batch]], toTorch=True)
	
	embed_words = MODEL.embeddings(sents[0])
	embeddings = MODEL.encoder(embed_words, lengths[0])
	return embeddings.cpu()

def perform_SentEval(model):
	global MODEL 
	MODEL = model
	# Set params for SentEval
	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
	params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
									 'tenacity': 3, 'epoch_size': 2}

	# Set up logger
	logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

	se = senteval.engine.SE(params_senteval, batcher, prepare)
	# transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
	# 					'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
	# 					'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
	# 					'Length', 'WordContent', 'Depth', 'TopConstituents',
	# 					'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
	# 					'OddManOut', 'CoordinationInversion']
	transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment', 'SICKRelatedness', 'STS14']
	results = se.eval(transfer_tasks)
	print(results)
	return results


if __name__ == "__main__":
	# Prepare argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str)
	args = parser.parse_args()

	model_type, model_params, _ = args_to_params(load_args(args.checkpoint_path))
	model = create_model(args.checkpoint_path, model_type, model_params)
	perform_SentEval(model)


