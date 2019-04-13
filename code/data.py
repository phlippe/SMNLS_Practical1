
import os
import numpy as np
import torch
import json
import re
import sys
from random import shuffle

# 0 => Full debug
# 1 => Reduced output
# 2 => No output at all (on cluster)
DEBUG_LEVEL = 0

def set_debug_level(level):
	global DEBUG_LEVEL
	DEBUG_LEVEL = level

def debug_level():
	global DEBUG_LEVEL
	return DEBUG_LEVEL

def build_vocab(word_list, glove_path='../glove.840B.300d.txt'):
	word2vec = {}
	num_ignored_words = 0
	num_missed_words = 0
	num_found_words = 0
	overall_num_words = len(word_list)
	with open(glove_path, "r") as f:
		lines = f.readlines()
		number_lines = len(lines)
		for i, line in enumerate(lines):
			if debug_level() == 0:
				print("Processed %4.2f%% of the glove (found %4.2f%% of words yet)" % (100.0 * i / number_lines, 100.0 * num_found_words / overall_num_words), end="\r")
			if num_found_words == overall_num_words:
				break
			# if num_found_words * 1.0 / overall_num_words > 0.7:
			# 	break
			word, vec = line.split(' ', 1)
			if word in word_list:
				glove_vec = [float(x) for x in vec.split()]
				word2vec[word] = np.array(glove_vec)
				num_found_words += 1
			else:
				num_ignored_words += 1

	example_missed_words = list()
	for word in word_list:
		if word not in word2vec:
			num_missed_words += 1
			if num_missed_words < 30:
				example_missed_words.append(word)

	print("Created vocabulary with %i words. %i words were ignored from Glove, %i words were not found in embeddings." % (len(word2vec.keys()), num_ignored_words, num_missed_words))
	if num_missed_words > 0:
		print("Example missed words: " + " +++ ".join(example_missed_words))

	return word2vec

def load_word2vec_from_file(word_file="small_glove_words.txt", numpy_file="small_glove_embed.npy"):
	word2vec = dict()
	word2id = dict()
	word_vecs = np.load(numpy_file)
	with open(word_file, "r") as f:
		for i, l in enumerate(f):
			word2vec[l.replace("\n","")] = word_vecs[i,:]
	index = 0
	for key, _ in word2vec.items():
		word2id[key] = index
		index += 1

	print("Loaded vocabulary of size " + str(word_vecs.shape[0]))
	return word2vec, word2id, word_vecs

def create_word2vec_vocab():
	val_dataset = SNLIDataset('dev')
	val_dataset.print_statistics()
	test_dataset = SNLIDataset('test')
	test_dataset.print_statistics()
	train_dataset = SNLIDataset('train')
	train_dataset.print_statistics()

	for dataset, name in zip([train_dataset, test_dataset, val_dataset], ["train", "test", "val"]):
		filename = name + "_word_list.txt"
		if not os.path.isfile(filename):
			word_list = val_dataset.get_word_list()
			with open(filename, "w") as f:
				f.write("\n".join(word_list))

	train_word_list = [l.rstrip() for l in open("train_word_list.txt", "r")]
	test_word_list = [l.rstrip() for l in open("test_word_list.txt", "r")]
	val_word_list = [l.rstrip() for l in open("val_word_list.txt", "r")]
	
	word_list = list(set(val_word_list + test_word_list + train_word_list)) + ['<s>', '</s>', '<p>', 'UNK']
	print("Number of unique words in all datasets: " + str(len(word_list)))
	voc = build_vocab(word_list)
	np_word_list = []
	with open('small_glove_words.txt', 'w') as f:
		# json.dump(voc, f)
		for key, val in voc.items():
			f.write(key + "\n")
			np_word_list.append(val)
	np_word_array = np.stack(np_word_list, axis=0)
	np.save('small_glove_embed.npy', np_word_array)

SNLI_TRAIN_DATASET = None
SNLI_VAL_DATASET = None
SNLI_TEST_DATASET = None
SNLI_WORD2VEC = None
SNLI_WORD2ID = None
SNLI_WORDVEC_TENSOR = None

def load_SNLI_datasets(debug_dataset=False):
	# Train dataset takes time to load. If we just want to shortly debug the pipeline, set "debug_dataset" to true. Then the validation dataset will be used for training
	global SNLI_TRAIN_DATASET, SNLI_VAL_DATASET, SNLI_TEST_DATASET, SNLI_WORD2VEC, SNLI_WORD2ID, SNLI_WORDVEC_TENSOR
	
	if SNLI_WORD2VEC is None or SNLI_WORD2ID is None or SNLI_WORDVEC_TENSOR is None:
		SNLI_WORD2VEC, SNLI_WORD2ID, SNLI_WORDVEC_TENSOR = load_word2vec_from_file()

	if SNLI_TRAIN_DATASET is None:
		train_dataset = SNLIDataset('train' if not debug_dataset else 'dev', shuffle_data=True)
		train_dataset.print_statistics()
		train_dataset.set_vocabulary(SNLI_WORD2ID)
		SNLI_TRAIN_DATASET = train_dataset

	if SNLI_VAL_DATASET is None:
		val_dataset = SNLIDataset('dev', shuffle_data=False)
		val_dataset.print_statistics()
		val_dataset.set_vocabulary(SNLI_WORD2ID)
		SNLI_VAL_DATASET = val_dataset

	if SNLI_TEST_DATASET is None:
		test_dataset = SNLIDataset('test', shuffle_data=False)
		test_dataset.print_statistics()
		test_dataset.set_vocabulary(SNLI_WORD2ID)
		SNLI_TEST_DATASET = test_dataset

	return SNLI_TRAIN_DATASET, SNLI_VAL_DATASET, SNLI_TEST_DATASET, SNLI_WORD2VEC, SNLI_WORD2ID, SNLI_WORDVEC_TENSOR



class SNLIDataset:

	# Data type either train, dev or test
	def __init__(self, data_type, data_path="../snli_1.0", add_suffix=True, shuffle_data=True):
		self.load_data(data_path, data_type)
		self.data_type = data_type
		self.example_index = 0
		self.perm_indices = list(range(1,len(self.data_list)))
		self.shuffle_data = shuffle_data
		if self.shuffle_data:
			shuffle(self.perm_indices)

	def load_data(self, data_path, data_type):
		self.data_list = list()
		self.num_invalids = 0
		s1 = [line.rstrip() for line in open(data_path + "/s1." + data_type, 'r')]
		s2 = [line.rstrip() for line in open(data_path + "/s2." + data_type, 'r')]
		labels = [NLIData.LABEL_LIST[line.rstrip('\n')] for line in open(data_path + "/labels." + data_type, 'r')]
		
		i = 0
		for prem, hyp, lab in zip(s1, s2, labels):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset" % (100.0 * i / len(s1)), end="\r")
			i += 1
			if lab == -1:
				self.num_invalids += 1
				continue
			d = NLIData(premise = prem, hypothesis = hyp, label = lab)
			self.data_list.append(d)

	def get_word_list(self):
		all_words = dict()
		for i, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Processed %4.2f%% of the dataset" % (100.0 * i / len(self.data_list)), end="\r")
			data_words = data.premise_words + data.hypothesis_words
			for w in data_words:
				if w not in all_words:
					all_words[w] = ''
		all_words = list(all_words.keys())
		print("Found " + str(len(all_words)) + " unique words")
		return all_words

	def set_vocabulary(self, word2vec):
		missing_words = 0
		overall_words = 0
		for data in self.data_list:
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec)
			missing_words += mw 
			overall_words += ow 
		print("Amount of missing words: %4.2f%%" % (100.0 * missing_words / overall_words))

	def print_statistics(self):
		print("="*50)
		print("Dataset statistics " + self.data_type)
		print("-"*50)
		print("Number of examples: " + str(len(self.data_list)))
		print("Labelwise amount:")
		for key, val in NLIData.LABEL_LIST.items():
			print("\t- " + key + ": " + str(sum([d.label == val for d in self.data_list])))
		print("Number of invalid examples: " + str(self.num_invalids))
		print("="*50)

	def _get_next_example(self):
		exmp = self.data_list[self.perm_indices[self.example_index]]
		self.example_index += 1
		if self.example_index >= len(self.perm_indices):
			if self.shuffle_data:
				shuffle(self.perm_indices)
			self.example_index = 0
		return exmp

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False, bidirectional=False):
		# Output sentences with dimensions (bsize, max_len)
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		batch_s1 = []
		batch_s2 = []
		batch_labels = []
		for _ in range(batch_size):
			data = self._get_next_example()
			batch_s1.append(data.premise_vocab)
			batch_s2.append(data.hypothesis_vocab)
			batch_labels.append(data.label)
			if bidirectional:
				batch_s1.append(data.premise_vocab[::-1])
				batch_s2.append(data.hypothesis_vocab[::-1])

		lengths = []
		embeds = []
		for batch_sents in [batch_s1, batch_s2]:
			lengths_sents = np.array([x.shape[0] for x in batch_sents])
			max_len = np.max(lengths_sents)
			sent_embeds = np.zeros((len(batch_sents), max_len), dtype=np.int32)
			for s_index, sent in enumerate(batch_sents):
				sent_embeds[s_index, :sent.shape[0]] = sent
			if toTorch:
				sent_embeds = torch.LongTensor(sent_embeds)
				lengths_sents = torch.LongTensor(lengths_sents)
				if torch.cuda.is_available():
					sent_embeds = sent_embeds.cuda()
					lengths_sents = lengths_sents.cuda()
			lengths.append(lengths_sents)
			embeds.append(sent_embeds)
		if toTorch:
			batch_labels = torch.LongTensor(np.array(batch_labels))
			if torch.cuda.is_available():
				batch_labels = batch_labels.cuda()
		return embeds, lengths, batch_labels

	def get_num_examples(self):
		return len(self.data_list)

	def get_num_classes(self):
		c = 0
		for key, val in NLIData.LABEL_LIST:
			if val >= 0:
				c += 1
		return c


class NLIData:

	LABEL_LIST = {
		'-': -1,
		"neutral": 0, 
		"entailment": 1,
		"contradiction": 2
	}

	def __init__(self, premise, hypothesis, label):
		self.premise_words = self._preprocess_sentence(premise)
		self.hypothesis_words = self._preprocess_sentence(hypothesis)
		self.premise_vocab = None
		self.hypothesis_vocab = None
		self.label = label

	def translate_to_dict(self, word_dict):
		self.premise_vocab = self._sentence_to_dict(word_dict, self.premise_words)
		self.hypothesis_vocab = self._sentence_to_dict(word_dict, self.hypothesis_words)

	def _preprocess_sentence(self, sent):
		sent_words = sent.lower().split(" ") 
		for i in range(len(sent_words)):
			if len(sent_words[i]) > 1 and "." in sent_words[i]:
				sent_words[i] = sent_words[i].replace(".","")
		return sent_words

	def _sentence_to_dict(self, word_dict, sent):
		vocab_words = list()
		vocab_words += [word_dict['<s>']]
		for w in sent:
			if w in word_dict:
				vocab_words.append(word_dict[w])
		vocab_words += [word_dict['</s>']]
		vocab_words = np.array(vocab_words, dtype=np.int32)
		return vocab_words

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		for w in (self.premise_words + self.hypothesis_words):
			if w not in word_dict:
				missing_words += 1
		return missing_words, (len(self.premise_words) + len(self.hypothesis_words))

		
	def get_data(self):
		return premise_vocab, hypothesis_vocab, label


if __name__ == '__main__':
	# create_word2vec_vocab()
	train_dataset, val_dataset, test_dataset, word2vec, word2id, wordvec_tensor = load_SNLI_datasets()
	embeds, lengths, batch_labels = train_dataset.get_batch(8)
	print("Embeddings: " + str(embeds))
	print("Lengths: " + str(lengths))
	print("Labels: " + str(batch_labels))


