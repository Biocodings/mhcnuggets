'''
Predict IC50s for a batch of peptides
using a trained model

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

from __future__ import print_function
import numpy as np
from models import get_predictions
import models
import dataset
from keras.optimizers import Adam
import argparse


def predict(model, weights_path, peptides_path):
    '''
    Training protocol
    '''

    # read peptides
    peptides = [p.strip() for p in open(peptides_path)]

    print('Predicting for %d peptides' % (len(peptides)))
    # apply cut/pad or mask to same length
    if 'lstm' in model or 'gru' in model:
        normed_peptides = dataset.mask_peptides(peptides)
    else:
        normed_peptides = dataset.cut_pad_peptides(peptides)

    # get tensorized values for prediction
    peptides_tensor = dataset.tensorize_keras(normed_peptides, embed_type='softhot')

    # make model
    print('Building model')
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

    model.load_weights(weights_path)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))

    # test model
    preds_continuous, preds_binary = get_predictions(peptides_tensor, model)
    ic50s = [dataset.map_proba_to_ic50(p[0]) for p in preds_continuous]
    for i, peptide in enumerate(peptides):
        print(peptide, ic50s[i])


def parse_args():
    '''
    Parse user arguments
    '''

    info = 'Predict IC50 for a batch of peptides using a trained model'
    parser = argparse.ArgumentParser(description=info)

    parser.add_argument('-m', '--model',
                        type=str, default='lstm',
                        help=('Type of MHCnuggets model used to predict' +
                              'options are fc, gru, lstm, chunky_cnn, ' +
                              'spanny_cnn'))

    parser.add_argument('-w', '--weights',
                        type=str, required=True,
                        help='Path to weights of the model')

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
    predict(opts['model'], opts['weights'], opts['peptides'])


if __name__ == '__main__':
    main()
