# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:07:50 2021

@author: bejen
"""


#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
    
    # Peptides:
    l1_pep_conv_1 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l1_pep_conv_k1')(x_pep)
    l1_pep_conv_3 = layers.Conv1D(n_filters, 3, padding='same', activation='sigmoid', name='l1_pep_conv_k3')(x_pep)
    l1_pep_conv_5 = layers.Conv1D(n_filters, 5, padding='same', activation='sigmoid', name='l1_pep_conv_k5')(x_pep)
    l1_pep_conv_7 = layers.Conv1D(n_filters, 7, padding='same', activation='sigmoid', name='l1_pep_conv_k7')(x_pep)
    l1_pep_conv_9 = layers.Conv1D(n_filters, 9, padding='same', activation='sigmoid', name='l1_pep_conv_k9')(x_pep)
    
    # Second convolutional layer
    p1_conc_t1 = layers.Concatenate(axis=1, name = 'pep_tcr_conc1')([ l1_pep_conv_1, l1_tcr_conv_1])
    p3_conc_t3 = layers.Concatenate(axis=1, name = 'pep_tcr_conc3')([ l1_pep_conv_3, l1_tcr_conv_3])
    p5_conc_t5 = layers.Concatenate(axis=1, name = 'pep_tcr_conc5')([ l1_pep_conv_5, l1_tcr_conv_5])
    p7_conc_t7 = layers.Concatenate(axis=1, name = 'pep_tcr_conc7')([ l1_pep_conv_7, l1_tcr_conv_7])
    p9_conc_t9 = layers.Concatenate(axis=1, name = 'pep_tcr_conc9')([ l1_pep_conv_9, l1_tcr_conv_9])

    l2_conv_1 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l2_conv_1')(p1_conc_t1)
    l2_conv_3 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l2_conv_3')(p3_conc_t3)
    l2_conv_5 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l2_conv_5')(p5_conc_t5)
    l2_conv_7 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l2_conv_7')(p7_conc_t7)
    l2_conv_9 = layers.Conv1D(n_filters, 1, padding='same', activation='sigmoid', name='l2_conv_9')(p9_conc_t9)
    
    #Max pooling
    
    l3_conv_1_max = tf.reduce_max(l2_conv_1, axis = 1)
    l3_conv_3_max = tf.reduce_max(l2_conv_3, axis = 1)
    l3_conv_5_max = tf.reduce_max(l2_conv_5, axis = 1)
    l3_conv_7_max = tf.reduce_max(l2_conv_7, axis = 1)
    l3_conv_9_max = tf.reduce_max(l2_conv_9, axis = 1)
    

    # Concatenate peptide layers

    l4_input_pep_tcr = layers.Flatten(name = 'concat_pep_tcr_flatten')(layers.Concatenate(axis=1, name = 'l3_pep_conv_concatenate_layers')([l3_conv_1_max, l3_conv_3_max, l3_conv_5_max,
                                           l3_conv_7_max, l3_conv_9_max]))

        
    
    # Dense Layer:    
    l_dense = layers.Dense(n_hidden, activation = 'sigmoid', use_bias = True, name = 'dense_layer_100_hidden_units')(l4_input_pep_tcr) 
    
    
    # For Dropout layer - uncomment this 
    l_dense_drop = layers.Dropout(0.2, noise_shape=None, seed=None, name = 'dropout_layer_0.2_droput_rate')(l_dense)
    
    output = layers.Dense(1, activation = 'sigmoid', name = 'Output_predicted')(l_dense_drop)
    
    
    return x_tcr, x_pep, output

#%%
# Uncomment this for a model summary and obtaining a graph of the CNN:
'''
x_tcr, x_pep, output = build_CNN(n_filters=100, n_hidden=100)

model = keras.Model(
    inputs=[x_tcr, x_pep],
    outputs=[output],
    name = 'NetTCR 1.0 - Keras Model Summary'
    )

model.summary()


# Drawing the CNN:

import tensorflow.keras.utils
from importlib import reload
reload(tensorflow.keras.utils)

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot_netTCR1.png', show_shapes=True, show_layer_names=True)
'''
