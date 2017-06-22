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
* MHCnuggets-Chunky-CNN
* MHCnuggets-Spanny-CNN

### Training ###
Training a model is simple. For example, to train a MHCnuggets-LSTM model
for 100 epochs on the Kim dataset for HLA-A*02:01 and save it to test/tmp.h5:
```bash
python scripts/train.py -a HLA-A0201 -s test/HLA-A0201.h5 -n 100 -m lstm -d data/kim2014/train.csv
```

### Transfer learning ###
Transfer learning is just as easy. For example, if we wanted to train the
a model for HLA-A*02:03 with weights that are learned from HLA-A*02:01 instead of
random initialization:
```bash
python scripts/train.py -a HLA-A0203 -s test/HLA-A0203.h5 -n 100 -m lstm -d data/kim2014/train.csv -t test/HLA-A0201.h5
```
Note that the model architectures used for transfer learning must be the same eg MHCnuggets-LSTM to MHCnuggets-LSTM.


### Predicting ###



### Requirements ###
* Keras w/ Theano or Tensorflow backend
* Numpy
* Scipy