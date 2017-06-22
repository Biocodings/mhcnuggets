'''
Implementation of MHCnuggets models in
Keras

Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
import keras.metrics
from keras.layers.core import Dropout, Flatten, Masking
from keras.layers.recurrent import LSTM, GRU
import math
from keras.layers import Input
from keras.layers import Conv1D, GlobalMaxPooling1D
from aa_embeddings import MASK_VALUE

# constants
IC50_THRESHOLD = 500
MAX_IC50 = 50000
MAX_CUT_LEN = 9
MAX_MASK_LEN = 11
NUM_AAS = 21


def get_predictions(test_peptides, model, binary=False, embed_peptides=None):
    '''
    Get predictions from a given model
    '''

    if embed_peptides is None:
        preds_cont = model.predict(test_peptides)
    else:
        preds_cont = model.predict([test_peptides, embed_peptides])
    preds_cont = [0 if y < 0 else y for y in preds_cont]
    if not binary:
        preds_bins = [1 if y >= 1-math.log(IC50_THRESHOLD, MAX_IC50)
                      else 0 for y in preds_cont]
    else:
        preds_bins = [1 if y >= 0.5 else 0 for y in preds_cont]
    return preds_cont, preds_bins


def mhcnuggets_fc(input_size=(MAX_CUT_LEN, NUM_AAS),
                  hidden_size=64, output_size=1, dropout=0.8):
    '''
    MHCnuggets-FC model
    -----------
    input_size : Num dimensions of the encoding (9, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    '''

    model = Sequential()
    model.add(Flatten(input_shape=input_size))
    model.add(Dense(hidden_size))
    model.add(Dropout(dropout))
    model.add(Activation('tanh'))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model


def mhcnuggets_lstm(input_size=(MAX_MASK_LEN, NUM_AAS),
                    hidden_size=64, output_size=1, dropout=0.2):
    '''
    MHCnuggets-LSTM model
    -----------
    input_size : Num dimensions of the encoding (11, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    '''

    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(input_size)))
    model.add(LSTM(hidden_size, dropout=dropout, recurrent_dropout=dropout))
    model.add(Dense(hidden_size))
    model.add(Dropout(dropout))
    model.add(Activation('tanh'))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model


def mhcnuggets_gru(input_size=(MAX_MASK_LEN, NUM_AAS),
                   hidden_size=64, output_size=1, dropout=0.2):
    '''
    MHCnuggets-GRU model
    -----------
    input_size : Num dimensions of the encoding (11, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    '''

    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(input_size)))
    model.add(GRU(hidden_size, dropout=dropout, recurrent_dropout=dropout, input_shape=(input_size)))
    model.add(Dense(hidden_size))
    model.add(Dropout(dropout))
    model.add(Activation('tanh'))
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))
    return model


def mhcnuggets_spanny_cnn(input_size=(MAX_CUT_LEN, NUM_AAS),
                          hidden_size=64, output_size=1,
                          dropout=0.6, n_filters=1):
    '''
    MHCnuggets-Spanny-CNN model
    -----------
    input_size : Num dimensions of the encoding (11, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    n_filters : number of filters
    '''

    main_input = Input(shape=input_size)
    conv_1 = Conv1D(filters=n_filters, kernel_size=2, padding='valid', activation='tanh',
                    strides=1)(main_input)
    max_1 = GlobalMaxPooling1D()(conv_1)
    conv_2 = Conv1D(filters=1, kernel_size=9, padding='valid', activation='tanh',
                    strides=1)(main_input)
    max_2 = GlobalMaxPooling1D()(conv_2)
    conv_3 = Conv1D(filters=n_filters, kernel_size=3, padding='valid', activation='tanh',
                    strides=1)(main_input)
    max_3 = GlobalMaxPooling1D()(conv_3)
    x = keras.layers.concatenate([max_1, max_2, max_3])
    x = Dense(hidden_size)(x)
    x = Dropout(dropout)(x)
    x = Activation('tanh')(x)
    x = Dense(output_size)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=[main_input], outputs=[x])
    return model


def mhcnuggets_chunky_cnn(input_size=(MAX_CUT_LEN, NUM_AAS),
                          hidden_size=64, output_size=1,
                          dropout=0.6, embed_size=16, n_filters=250):
    '''
    MHCnuggets-Chunky-CNN model
    -----------
    input_size : Num dimensions of the encoding (11, 21) = (len, numAA)
    hidden_size : Num hidden dimensions
    output_size : Num of outputs
    dropout : dropout probability to apply
    n_filters : number of filters
    '''

    main_input = Input(shape=input_size)
    conv_1 = Conv1D(filters=n_filters, kernel_size=2, padding='valid', activation='tanh',
                    strides=1)(main_input)
    max_1 = GlobalMaxPooling1D()(conv_1)
    conv_2 = Conv1D(filters=n_filters, kernel_size=3, padding='valid', activation='tanh',
                    strides=1)(main_input)
    max_2 = GlobalMaxPooling1D()(conv_2)
    x = keras.layers.concatenate([max_1, max_2])
    x = Dense(hidden_size)(x)
    x = Dropout(dropout)(x)
    x = Activation('tanh')(x)
    x = Dense(output_size)(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=[main_input], outputs=[x])
    return model
