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

Training a model is simple. For example, to train a MHCnuggets-LSTM model
for 100 epochs on the Kim dataset for HLA-A*02:03 and save it to test/tmp.h5:
```bash
python scripts/train.py -a HLA-A0203 -s test/tmp.h5 -n 100 -m lstm -d data/kim2014/train.csv
```


### Requirements ###
* Keras w/ Theano or Tensorflow backend
* Numpy
* Scipy