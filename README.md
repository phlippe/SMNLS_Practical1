# SMNLS Practical 1
* _Course_: Stochastical Methods in Natural Language Semantics
* _Author_: Phillip Lippe

## Preparation
Requirements file to install all needed packages


## Functionality

### Data pre-processing

### Training

1. **Model parameters** 

	The model is configured by an encoder and a classifier.

	1. *Encoder*

		Currently, four different encoder architectures are implemented. They can be chosen by the option ```--model```:

		* ```--model 0```: Bow Baseline
		* ```--model 1```: Uni-directional LSTM
		* ```--model 2```: Bi-directional LSTM
		* ```--model 3```: Bi-directional LSTM with max pooling

		Further options for the encoder are:

		* ```--embed_dim VAL```: The embedding dimensionality of the sentence (output of the encoder). Will be automatically set to 300 in the case of the BoW model (equals to word dimensionality). Default: 2048

	2. *Classifier*

		The default classifier consists of a MLP with two linear layers. The hidden dimensionality is 512, and the output are 3 classes. 

		* ```--fc_dim VAL```: The number of hidden units in the classifier. Default: 512
		* ```--fc_dropout VAL```: Dropout applied on the hidden layer. Default: 0.0, no dropout
		* ```--fc_nonlinear```: If selected, a non-linear tanh activation is applied between the two linear layers.

2. **Optimizer parameters**

	Two different optimizers are supported:

	* **SGD** (```--optimizer 0```). The vanilla SGD optimizer with additional parameters:

		* ```--weight_decay VAL```: Applies weight decay by adding a l2 loss over weights. Default: 1e-2 (value in the paper)
		* ```--momentum VAL```: Applies momentum. Default: 0.0 (no momentum)

	* **Adam** (```--optimizer 1```). Adam optimizer with default parameter values as specified by PyTorch optimizer package.

	Additional parameters valid for every optimizer:

	* ```--learning_rate VAL```: Learning rate. Default: 0.1
	* ```--lr_decay VAL```: The learning rate is reduced after every epoch if the evaluation accuracy is worse than the mean of the two previous. If the option ```--intermediate_evals```is activated, then the average of the intermediate evaluations is compared to the pre-last epoch accuracy. The reduction factor can be specified by this option. Default: 0.2
	* ```--lr_max_red_steps VAL```: After how many learning rate reduction steps the training automatically stops. Default: 5
	* ```--batch_size VAL```: Batch size used during training. Default: 64

3. **Training configuration**
	
	Parameters specifying training configuration:

	* ```--checkpoint_path VAL```: The path where the checkpoints should be saved. Default: a new folder in "checkpoints/" will be created based on the date and time.
	* ```--restart```: If checkpoint folder is not empty, the previously stored checkpoints etc. are deleted. 
	* ```--load_config```: If checkpoint folder is not empty, the configuration saved in this folder (from the previous experiment) will be loaded. Recommended if experiment should be continued.
	* ```--tensorboard```: If selected, a tensorboard is created during training. Stored in the same folder as the checkpoints.
	* ```--intermediate_evals```: If selected, a evaluation will be performed every 2000 iterations. Gives a less noisy estimate of the evaluation accuracy. 
	* ```--seed VAL```: What seed to set to guarantee reproducability of experiment. Default: 42. 
	* ```--cluster```: Strongly recommended if the experiments are executed on the clusters. Reduces the output on the terminal.
	* ```--debug```: If selected, the evaluation set will be used for training. _ONLY_ use this option for debugging of the model.


### Evaluation

### Inference

### Examples/Experiments

* Training Baseline
```
python train.py --model 0 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/Baseline
```

* Training uni-directional LSTM 
```
python train.py --model 1 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/LSTM
```

* Training bidirectional LSTM 
```
python train.py --model 2 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/BiLSTM
```

* Training Bi-LSTM with max pooling
```
python train.py --model 3 --optimizer 0 --learning_rate 0.1 --weight_decay 1e-5 --lr_max_red_steps 5 --intermediate_evals --checkpoint_path checkpoints/BiLSTM_Max
```

## Results/Visualization

### Visualization

## TODO
* Add inference file
* Add weight distribution to tensorboard
* Add sentence embedding distribution to tensorboard
* LSTM vs Baseline
	* Boring baseline
	* LSTM able to learn relations between words (e.g. "not raining")
	* But still, baseline performs quite well (2/3 correct)? Why? => check bias of dataset
* Bi-LSTM
	* Find sentences on which it performs significantly better than standard LSTM
* Bi-LSTM max pooling
	* Add visualization of importance of words in a sentence for max pooling
	* Analyse what words activate which feature channels the most
	* Idea to test out: drop a few channels of the sentence embedding and look again on TSNE
* General
	* TSNE for sentence embeddings on various datasets (labels given by e.g. SentEval tasks)
	* Test for bias of dataset (augment premise)
* 