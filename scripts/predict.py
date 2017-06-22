'''
Training a GRU model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
from dataset import Dataset
import dataset
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
WEIGHTS_DIR = 'saves/gru_finetune_weights/'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.tsv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
MAX_LEN = 11
NUM_AAS = 21  # including pad X
# NUM_EPOCH = 200
NUM_EPOCH = 25
NUM_HIDDEN = 64
BATCH_SIZE = 32
LR = 0.001
BETA = 0.9
np.random.seed(1437)  # for reproducibility


def train(mhc, pepfile):
    '''
    Training protocol
    '''

    # read peptides
    peptides = [p.strip() for p in open(pepfile)]

    print('Predicting for %d peptides binding to MHC %s' % (len(peptides), mhc))

    # define the path to save weights
    model_path = os.path.join(WEIGHTS_DIR, mhc + '.h5')

    # make model
    print('Building model')
    model = keras_models.GRU_(input_size=(MAX_LEN, NUM_AAS))
    model.load_weights(model_path)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # get tensorized values for testing
    masked_peptides = dataset.mask_peptides(peptides)
    peptides_tensor = dataset.tensorize_keras(masked_peptides, embed_type='softhot')

    # test model
    preds_continuous, preds_binary = get_predictions(peptides_tensor, model)

    print('Done')

def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Test an RNN on an MHC from kim 2014 data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--mhc',
                        type=str, required=True,
                        help='MHC molecule to train on')

    parser.add_argument('-p', '--peptides',
                        type=str, required=True,
                        help='New line separated list of peptides')

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    train(opts['mhc'], opts['peptides'])


if __name__ == '__main__':
    main()
