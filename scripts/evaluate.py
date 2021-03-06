'''
Script to evaluate a MHCnuggets model
of choice

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

# imports
from __future__ import print_function
from dataset import Dataset
import numpy as np
import os
from models import get_predictions
import models
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import kendalltau
from keras.optimizers import Adam
import argparse


def test(mhc, data, model, model_path):
    '''
    Training protocol
    '''

    # print out options
    print('Testing\nMHC: %s\nData: %s\nModel: %s\nSave path: %s' %
          (mhc, data, model, model_path))

    # load training
    test_data = Dataset.from_csv(filename=data,
                                 sep=',',
                                 allele_column_name='mhc',
                                 peptide_column_name='peptide',
                                 affinity_column_name='IC50(nM)')

    # apply cut/pad or mask to same length
    if 'lstm' in model or 'gru' in model:
        test_data.mask_peptides()
    else:
        test_data.cut_pad_peptides()

    # get the allele specific data
    mhc_test = test_data.get_allele(mhc)

    # define model
    if model == 'fc':
        model = models.mhcnuggets_fc()
    elif model == 'gru':
        model = models.mhcnuggets_gru()
    elif model == 'lstm':
        model = models.mhcnuggets_lstm()
    elif model == 'chunky_cnn':
        model = models.mhcnuggets_chunky_cnn()
    elif model == 'spanny_cnn':
        model = models.mhcnuggets_spanny_cnn()

    # compile model
    model.load_weights(model_path)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # get tensorized values for training
    test_peptides, test_continuous, test_binary = mhc_test.tensorize_keras(embed_type='softhot')

    # test
    preds_continuous, preds_binary = get_predictions(test_peptides, model)
    test_auc = roc_auc_score(test_binary, preds_continuous)
    test_f1 = f1_score(test_binary, preds_binary)
    test_ktau = kendalltau(test_continuous, preds_continuous)[0]
    print('Test AUC: %.4f, F1: %.4f, KTAU: %.4f' %
          (test_auc, test_f1, test_ktau))


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Train a MHCnuggets model on given data'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-d', '--data',
                        type=str, default='data/kim2014/test.csv',
                        help=('Data file to use for training, should ' +
                              'be csv files w/ formatting similar to ' +
                              'data/kim2014/test.csv'))

    parser.add_argument('-a', '--allele',
                        type=str, required=True,
                        help='MHC allele to evaluate on')

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help=('Type of MHCnuggets model to evaluate' +
                              'options are fc, gru, lstm, chunky_cnn, ' +
                              'spanny_cnn'))

    parser.add_argument('-s', '--save_path',
                        type=str, required=True,
                        help=('Path to which the model weights are saved'))

    args = parser.parse_args()
    return vars(args)


def main():
    '''
    Main function
    '''

    opts = parse_args()
    test(opts['allele'], opts['data'], opts['model'], opts['save_path'])


if __name__ == '__main__':
    main()
