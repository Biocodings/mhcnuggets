# MHCnuggets 1.0

This repository contains scripts for training and making
predictions using the MHCnuggets models! It also
contains the pre-trained models for the Kim 2014 benchmark
as seen in the MHCnuggets paper (link). Production models
trained on the latest IEDB data are on the way.

### Currently available models ###
* MHCnuggets-LSTM
* MHCnuggets-GRU
* MHCnuggets-FC
* MHCnuggets-Spanny-CNN
* MHCnuggets-Chunky-CNN

### Which model should I use? ###
Here's a table to help you decide! (evaluation on the Kim 2014 benchmark)
Our personal recommendation is either the MHCnuggets-GRU or MHCnuggets-LSTM.
MHCnuggets-FC is fine too but MHCnuggets-Chunky-CNN and MHCnuggets-Spanny-CNN
are definitely a tier below the rest.

Method                 | AUC   | F1    | K-Tau
-----------------------| ----  | ------| -----
MHCnuggets-GRU         | 0.931 | 0.810 | 0.589
MHCnuggets-LSTM        | 0.931 | 0.806 | 0.587
MHCnuggets-FC          | 0.931 | 0.814 | 0.581
MHCnuggets-Spanny-CNN  | 0.918 | 0.795 | 0.563
MHCnuggets-Chunky-CNN  | 0.845 | 0.640 | 0.447


### Training ###
Training a model is simple. For example, to train a MHCnuggets-LSTM model
for 100 epochs on the Kim dataset for HLA-A\*02:01 and save it to test/tmp.h5:
```bash
python scripts/train.py -a HLA-A0201 -s test/HLA-A0201.h5 -n 100 -m lstm -d data/kim2014/train.csv
```

### Transfer learning ###
Transfer learning is just as easy. For example, if we wanted to train the
a model for HLA-A\*02:03 with weights that are learned from HLA-A\*02:01 instead of
random initialization:
```bash
python scripts/train.py -a HLA-A0203 -s test/HLA-A0203.h5 -n 25 -m lstm -d data/kim2014/train.csv -t test/HLA-A0201.h5
```
Note that the model architectures used for transfer learning must be the same e.g. MHCnuggets-LSTM to MHCnuggets-LSTM.
You also probably don't need as many epochs as when you're training from scratch, convergence with transfer learning
is usually a lot faster.

### Predicting ###
In order to predict for a set of peptides, provide the model name, the corresponding paths
to the trained weights, and the file containing new line separated peptides:
```bash
python scripts/predict.py -m lstm -w saves/kim2014/mhcnuggets_lstm/HLA-A0203.h5 -p test/test_peptides.peps
```

### Evaluation ###
Evaluation of the performance of a model on a dataset (Kim 2014 here) can be performed by:
```bash
python scripts/evaluate.py -a HLA-A0201 -m lstm -s saves/kim2014/mhcnuggets_lstm/HLA-A0201.h5 -d data/kim2014/test.csv
```

### Requirements ###
* Keras w/ Theano or Tensorflow backend
* Numpy
* Scipy