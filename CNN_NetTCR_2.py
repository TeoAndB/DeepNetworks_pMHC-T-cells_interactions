# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:55:21 2021

@author: bejen
"""
#%%
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

#%%

def build_CNN(n_filters=None, n_hidden=None): # In case of drop-rate add: drop_rate=None
    

    x_tcr = keras.Input(shape=(20,20), dtype='float32', name='TCRb') # instantiate a Keras tensor
    x_pep = keras.Input(shape=(9,20), dtype='float32', name='pep') # instantiate a Keras tensor
    
    # TCRs:
    l1_tcr_conv_1 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l1_tcr_conv_k1')(x_tcr)
    l1_tcr_conv_3 = layers.Conv1D(n_filters, 3, padding='same', activation='sigmoid', name='l1_tcr_conv_k3')(x_tcr)
    l1_tcr_conv_5 = layers.Conv1D(n_filters, 5, padding='same', activation='sigmoid', name='l1_tcr_conv_k5')(x_tcr)
    l1_tcr_conv_7 = layers.Conv1D(n_filters, 7, padding='same', activation='sigmoid', name='l1_tcr_conv_k7')(x_tcr)
    l1_tcr_conv_9 = layers.Conv1D(n_filters, 9, padding='same', activation='sigmoid', name='l1_tcr_conv_k9')(x_tcr)
    
    # Max Pooling - TCR
    l2_tcr_conv_1_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_tcr_maxpooling_k1')(l1_tcr_conv_1)
    l2_tcr_conv_3_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_tcr_maxpooling_k3')(l1_tcr_conv_3)
    l2_tcr_conv_5_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_tcr_maxpooling_k5')(l1_tcr_conv_5)
    l2_tcr_conv_7_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_tcr_maxpooling_k7')(l1_tcr_conv_7)
    l2_tcr_conv_9_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_tcr_maxpooling_k9')(l1_tcr_conv_9)
    
    
    # Concatenate TCR layers
    concat_tcr_l1 = layers.Flatten(name = 'tcr_flatten')(layers.Concatenate(axis=1, name = 'x_tcr_conv_concatenate_layers')([l2_tcr_conv_1_max, l2_tcr_conv_3_max, l2_tcr_conv_5_max,
                                           l2_tcr_conv_7_max, l2_tcr_conv_9_max]))
    
    # Peptides:
    l1_pep_conv_1 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l1_pep_conv_k1')(x_pep)
    l1_pep_conv_3 = layers.Conv1D(n_filters, 3, padding='same', activation='sigmoid', name='l1_pep_conv_k3')(x_pep)
    l1_pep_conv_5 = layers.Conv1D(n_filters, 5, padding='same', activation='sigmoid', name='l1_pep_conv_k5')(x_pep)
    l1_pep_conv_7 = layers.Conv1D(n_filters, 7, padding='same', activation='sigmoid', name='l1_pep_conv_k7')(x_pep)
    l1_pep_conv_9 = layers.Conv1D(n_filters, 9, padding='same', activation='sigmoid', name='l1_pep_conv_k9')(x_pep)
    
    # Max Pooling - peptide
    l2_pep_conv_1_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_pep_maxpooling_k1')(l1_pep_conv_1)
    l2_pep_conv_3_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_pep_maxpooling_k3')(l1_pep_conv_3)
    l2_pep_conv_5_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_pep_maxpooling_k5')(l1_pep_conv_5)
    l2_pep_conv_7_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_pep_maxpooling_k7')(l1_pep_conv_7)
    l2_pep_conv_9_max = layers.MaxPooling1D(pool_size=n_filters, padding = 'same', data_format='channels_first', name='l2_pep_maxpooling_k9')(l1_pep_conv_9)
    
    # Concatenate peptide layers
    concat_pep_l1 = layers.Flatten(name = 'concat_pep_flatten')(layers.Concatenate(axis=1, name = 'l2_pep_conv_concatenate_layers')([l2_pep_conv_1_max, l2_pep_conv_3_max, l2_pep_conv_5_max,
                                           l2_pep_conv_7_max, l2_pep_conv_9_max]))
    
    # Concatenate TCRb and peptide layers
    input_l1 = layers.Concatenate(axis=1, name = 'output_pep_tcr_concatenate')([concat_tcr_l1, concat_pep_l1])
    
    # Dense Layer:    
    l_dense = layers.Dense(n_hidden, activation = 'sigmoid', use_bias = True, name = 'dense_layer_100_hidden_units')(input_l1) 
    
    
    # For Dropout layer - uncomment this 
    #l_dense_drop = layers.Dropout(drop_rate, noise_shape=None, seed=None, name = 'dropout_layer_0.2_droput_rate')(l_dense)
    
    output = layers.Dense(1, activation = 'sigmoid', name = 'Output_predicted')(l_dense)
    
    
    return x_tcr, x_pep, output

#%%
# Uncomment this for a model summary and obtaining a graph of the CNN:
'''
x_tcr, x_pep, output = build_CNN(n_filters=16, n_hidden=100)

model = keras.Model(
    inputs=[x_tcr, x_pep],
    outputs=[output],
    name = 'NetTCR 2.0'
    )

model.summary()


# Drawing the CNN

import tensorflow.keras.utils
from importlib import reload
reload(tensorflow.keras.utils)

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot_netTCR2.png', show_shapes=True, show_layer_names=True)
'''
