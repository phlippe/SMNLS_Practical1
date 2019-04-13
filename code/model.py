import torch
import torch.nn as nn
import numpy as np 
import sys
import math


class NLIModel(nn.Module):

	AVERAGE_WORD_VECS = 0
	LSTM = 1
	BILSTM = 2
	BILSTM_MAX = 3

	def __init__(self, model_type, model_params, wordvec_tensor):
		super(NLIModel, self).__init__()

		self.embeddings = nn.Embedding(wordvec_tensor.shape[0], wordvec_tensor.shape[1])
		with torch.no_grad():
			self.embeddings.weight.data.copy_(torch.from_numpy(wordvec_tensor))
			self.embeddings.weight.requires_grad = False

		self.model_type = model_type
		self.model_params = model_params
		self._choose_encoder(model_type, model_params)
		self.classifier = NLIClassifier(model_params)

		if torch.cuda.is_available():
			self.embeddings = self.embeddings.cuda()
			self.encoder_1 = self.encoder_1.cuda()
			self.encoder_2 = self.encoder_2.cuda()
			self.classifier = self.classifier.cuda()

	def _choose_encoder(self, model_type, model_params):
		if model_type == NLIModel.AVERAGE_WORD_VECS:
			self.encoder_1 = EncoderBOW()
			self.encoder_2 = EncoderBOW()
		elif model_type == NLIModel.LSTM:
			self.encoder_1 = EncoderLSTM(model_params)
			self.encoder_2 = EncoderLSTM(model_params)
		elif model_type == NLIModel.BILSTM:
			self.encoder_1 = EncoderBILSTM(model_params)
			self.encoder_2 = EncoderBILSTM(model_params)
		elif model_type == NLIModel.BILSTM_MAX:
			self.encoder_1 = EncoderBILSTMPool(model_params)
			self.encoder_2 = EncoderBILSTMPool(model_params)
		else:
			print("Unknown encoder: " + str(model_type))
			sys.exit(1)


	def forward(self, words_s1, lengths_s1, words_s2, lengths_s2, applySoftmax=False):
		# Input must be [batch, time]
		embed_words_s1 = self.embeddings(words_s1)
		embed_words_s2 = self.embeddings(words_s2)

		embed_s1 = self.encoder_1(embed_words_s1, lengths_s1)
		embed_s2 = self.encoder_2(embed_words_s2, lengths_s2)

		out = self.classifier(embed_s1, embed_s2, applySoftmax=applySoftmax)
		return out


	def is_bidirectional(self):
		if self.model_type in [NLIModel.BILSTM, NLIModel.BILSTM_MAX]:
			return True
		return False



class NLIClassifier(nn.Module):

	def __init__(self, model_params):
		super(NLIClassifier, self).__init__()
		embed_sent_dim = model_params["embed_sent_dim"]
		fc_dropout = model_params["fc_dropout"] 
		fc_dim = model_params["fc_dim"]
		n_classes = model_params["n_classes"]

		input_dim = 4 * embed_sent_dim
		self.classifier = nn.Sequential(
			nn.Dropout(p=fc_dropout),
			nn.Linear(input_dim, fc_dim),
			nn.Tanh(),
			# nn.Dropout(p=fc_dropout),
			# nn.Linear(fc_dim, fc_dim),
			# nn.Tanh(),
			nn.Dropout(p=fc_dropout),
			nn.Linear(fc_dim, n_classes)
		)
		self.softmax_layer = nn.Softmax(dim=-1)

	def forward(self, embed_s1, embed_s2, applySoftmax=False):
		input_features = torch.cat((embed_s1, embed_s2, 
									torch.abs(embed_s1 - embed_s2), 
									embed_s1 * embed_s2), dim=1)
		# print("Input features: " + str(input_features.shape))
		out = self.classifier(input_features)
		if applySoftmax:
			out = self.softmax_layer(out)
		# print("Mean first dimensions: " + str(torch.mean(input_features[:,0:4], dim=0)))
		# print("Variance first dimensions: " + str(torch.var(input_features[:,0:4], dim=0)))
		# print("Values out: " + str(out[0:2,:]))
		
		return out

####################
## ENCODER MODELS ##
####################

class EncoderBOW(nn.Module):

	def __init__(self):
		super(EncoderBOW, self).__init__()

	def forward(self, embed_words, lengths):
		# Embeds are of shape [batch, time, embed_dim]
		# Lengths is of shape [batch]
		word_positions = torch.arange(start=0, end=embed_words.shape[1], dtype=lengths.dtype, device=embed_words.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		# X = torch.nn.utils.rnn.pack_padded_sequence(embed_words, lengths, batch_first=True)
		out = torch.sum(mask * embed_words, dim=1) / lengths.reshape([-1, 1]).float()
		return out


class EncoderLSTM(nn.Module):

	def __init__(self, model_params):
		super(EncoderLSTM, self).__init__()
		self.lstm_chain = LSTMChain(input_size=model_params["embed_word_dim"], 
									hidden_size=model_params["embed_sent_dim"])

	def forward(self, embed_words, lengths):
		final_states, _ = self.lstm_chain(embed_words, lengths)
		return final_states


class EncoderBILSTM(nn.Module):

	def __init__(self, model_params):
		super(EncoderBILSTM, self).__init__()
		self.lstm_chain = LSTMChain(input_size=model_params["embed_word_dim"], 
									hidden_size=int(model_params["embed_sent_dim"]/2))

	def forward(self, embed_words, lengths):
		# embed words is of shape [batch_size * 2, time, word_dim]
		final_states, _ = self.lstm_chain(embed_words, lengths)
		# Reshape to [batch_size, sent_dim]
		final_states = final_states.reshape(shape=[-1, 2 * final_states.shape[1]])
		return final_states


class EncoderBILSTMPool(nn.Module):

	def __init__(self, model_params):
		super(EncoderBILSTMPool, self).__init__()
		self.lstm_chain = LSTMChain(input_size=model_params["embed_word_dim"], 
									hidden_size=int(model_params["embed_sent_dim"]/2))

	def forward(self, embed_words, lengths):
		# embed words is of shape [batch_size * 2, time, word_dim]
		_, outputs = self.lstm_chain(embed_words, lengths)
		# Max time pooling
		pooled_features = EncoderBILSTMPool.pool_over_time(outputs, lengths)
		# Reshape to [batch_size, sent_dim]
		pooled_features = pooled_features.reshape(shape=[-1, 2 * pooled_features.shape[1]])
		return pooled_features

	@staticmethod
	def pool_over_time(outputs, lengths):
		time_dim = outputs.shape[1]
		word_positions = torch.arange(start=0, end=time_dim, dtype=lengths.dtype, device=outputs.device)
		mask = (word_positions.reshape(shape=[1, -1, 1]) < lengths.reshape([-1, 1, 1])).float()
		outputs = outputs * mask + (torch.min(outputs) - 1) * (1 - mask)
		final_states, _ = torch.max(outputs, dim=1)
		return final_states


		



###################################
## LOW LEVEL LSTM IMPLEMENTATION ##
###################################

class LSTMCell(nn.Module):

	def __init__(self, input_size, hidden_size, bias=True):
		"""Creates the weights for this LSTM"""
		super(LSTMCell, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.tanh_act = nn.Tanh()
		self.sigmoid_act = nn.Sigmoid()
		self.combined_gate = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

		self.reset_parameters()

	def reset_parameters(self):
		# Pytorch default initialization for LSTM weights
		stdv = 1.0 / math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)  

	def forward(self, input_, hx):
		"""
		input is (batch, input_size)
		hx is ((batch, hidden_size), (batch, hidden_size))
		"""
		prev_h, prev_c = hx

		# project input and prev state
		cat_input = torch.cat([input_,prev_h], dim=1)

		# main LSTM computation    
		combined_output = self.combined_gate(cat_input)
		i = self.sigmoid_act(combined_output[:,0 * self.hidden_size:1 * self.hidden_size])
		f = self.sigmoid_act(combined_output[:,1 * self.hidden_size:2 * self.hidden_size])
		g = self.tanh_act(combined_output[:,2 * self.hidden_size:3 * self.hidden_size])
		o = self.sigmoid_act(combined_output[:,3 * self.hidden_size:4 * self.hidden_size])

		c = f * prev_c + i * g
		h = o * self.tanh_act(c)

		return h, c

	def __repr__(self):
		return "{}({:d}, {:d})".format(
		self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMChain(nn.Module):

	def __init__(self, input_size, hidden_size, lstm_cell = None):
		super(LSTMChain, self).__init__()

		if lstm_cell is None:
			self.lstm_cell = LSTMCell(input_size, hidden_size)
		else:
			self.lstm_cell = lstm_cell


	def forward(self, word_embeds, lengths):
		batch_size = word_embeds.shape[0]
		time_dim = word_embeds.shape[1]
		embed_dim = word_embeds.shape[2]
		# Prepare default initial states
		hx = word_embeds.new_zeros(batch_size, self.lstm_cell.hidden_size)
		cx = word_embeds.new_zeros(batch_size, self.lstm_cell.hidden_size)
		# Iterate over time steps

		outputs = []   
		for i in range(time_dim):
			hx, cx = self.lstm_cell(word_embeds[:, i], (hx, cx))
			outputs.append(hx)

		# Stack output over time dimension => Output states per time step
		outputs = torch.stack(outputs, dim=0) 
		outputs = outputs.transpose(0, 1).contiguous()

		# Get final hidden state per sentence
		reshaped_outputs = outputs.view(-1, self.lstm_cell.hidden_size) # Reshape for better indexing
		indexes = (lengths - 1) + torch.arange(batch_size, device=word_embeds.device, dtype=lengths.dtype) * time_dim # Index of final states
		final = reshaped_outputs[indexes,:] # Final states
		# print("Final: " + str(final[:,0]))
		# print("="*50)
		# print("Input: " + str(word_embeds[0:2,:,0]))
		# print("-"*50)
		# print("Length: " + str(lengths[0:2]))
		# print("-"*50)
		# print("Output: " + str(outputs[0:2,:,0]))
		# print("-"*50)
		# print("Final: " + str(final[0:2,0]))
		# print("="*50)
		# for weight in self.lstm_cell.parameters():
		# 	if len(weight.shape) == 1:
		# 		print("Bias: " + str(weight[0]))
		# 	else:
		# 		print("Weight: " + str(weight[0,0]))
		return final, outputs



class ModuleTests():

	def __init__(self):
		pass

	def testEncoderBOW(self):
		enc = EncoderBOW()
		input_embeds = torch.ones((4, 16, 4))
		lengths = torch.Tensor(np.array([16, 8, 4, 2]))
		out = enc(input_embeds, lengths)
		print("Result: " + str(out))


	def testBiLSTMReshaping(self):
		output = torch.FloatTensor(np.array([[1, 3], [-2, -2], [2, -4], [-6, -1], [0, 0], [0, 0], [1, 2], [3, 4]], dtype=np.float32))
		output = output.reshape(shape=[-1, 2 * output.shape[1]])
		print("Result: " + str(output))

	def testTimePooling(self):
		output = torch.FloatTensor(np.array([[[1, 3], [-2, -2], [5, 1]], [[2, -4], [-6, -1], [0, 0]]], dtype=np.float32))
		lengths = torch.LongTensor(np.array([2, 3]))
		res = EncoderBILSTMPool.pool_over_time(output, lengths)
		print("Result: " + str(res))


if __name__ == '__main__':
	print(torch.__version__)
	tester = ModuleTests()
	tester.testEncoderBOW()
	tester.testTimePooling()
	tester.testBiLSTMReshaping()






