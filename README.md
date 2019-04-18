# SMNLS Practical 1
* _Course_: Stochastical Methods in Natural Language Semantics
* _Author_: Phillip Lippe

## TODO:
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