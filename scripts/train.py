'''
Training a GRU model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
from dataset import Dataset
import numpy as np
import os
from keras_models import get_predictions
import keras_models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from keras.optimizers import Adam, SGD
import argparse


# various paths/constants probably need to be moved
# somewhere more useful
DATA_DIR = 'data/kim2014'
WEIGHTS_DIR = 'saves/gru_vanilla_weights/'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.tsv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
MAX_LEN = 11
NUM_AAS = 21  # including pad X
NUM_EPOCH = 200
# NUM_EPOCH = 25
NUM_HIDDEN = 64
BATCH_SIZE = 32
LR = 0.001
BETA = 0.9
np.random.seed(1437)  # for reproducibility


def train(mhc):
    '''
    Training protocol
    '''

    # load training and testing data
    train_data = Dataset.from_csv(filename=TRAIN_PATH,
                                  sep='\t',
                                  allele_column_name='mhc',
                                  peptide_column_name='sequence',
                                  affinity_column_name='meas')

    # apply masking to same length
    train_data.mask_peptides()

    print('Training', mhc)

    # define the path to save weights
    model_path = os.path.join(WEIGHTS_DIR, mhc + '.h5')
    # pre_load_path = os.path.join(WEIGHTS_DIR, 'HLA-A0203.h5')

    # get the allele specific data
    mhc_train = train_data.get_allele(mhc)

    # make model
    model = keras_models.GRU_(input_size=(MAX_LEN, NUM_AAS))
    # model.load_weights(pre_load_path)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # get tensorized values for testing/
    train_peptides, train_continuous, train_binary = mhc_train.tensorize_keras(embed_type='softhot')

    # convergence criterion
    highest_f1 = -1
    lowest_bc = 1000

    for epoch in range(NUM_EPOCH):

        # train
        model.fit(train_peptides, train_continuous, epochs=1, verbose=0)
        # test model
        train_preds_cont, train_preds_bin = get_predictions(train_peptides, model)
        train_auc = roc_auc_score(train_binary, train_preds_cont)
        train_f1 = f1_score(train_binary, train_preds_bin)
        train_ktau = kendalltau(train_continuous, train_preds_cont)[0]
        print('epoch %d / %d' % (epoch, NUM_EPOCH))
        print('AUC: %.4f, F1: %.4f, KTAU: %.4f' % (train_auc, train_f1, train_ktau))

        # convergence
        if train_f1 > highest_f1:

            highest_f1 = train_f1
            best_epoch = epoch
            model.save_weights(model_path)


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Train an RNN on an MHC from kim 2014 data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--mhc',
                        type=str, required=True,
                        help='MHC molecule to train on')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    train(opts['mhc'])


if __name__ == '__main__':
    main()
