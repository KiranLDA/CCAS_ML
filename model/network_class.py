#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:39:58 2020

@author: kiran
"""


import datetime
import pickle

# from network.network_train import NetworkTrain
import numpy as np
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
from keras.layers import Masking
# from keras.layers import boolean_mask


class BuildNetwork():
    
    def __init__(self, x_train, num_calltypes, filters, gru_units, dense_neurons, dropout, mask_value):
        self.x_train = x_train
        self.num_calltypes = num_calltypes
        self.filters = filters
        self.gru_units = gru_units
        self.dense_neurons = dense_neurons
        self.dropout = dropout
        self.mask_value = mask_value
        
    def build_rnn_calltype(self):
        
        inp = Input(shape=(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3]))

        # Convolutional layers (conv - maxpool x3 )
        c_1 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(inp)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)        
        
        # reshape
        reshape_1 = Reshape((self.x_train.shape[-3], -1))(mp_3)
        
        # bidirectional gated recurrent unit x2
        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(reshape_1)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)
        
        # 3x relu
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=self.dropout)(dense_3)
        
        # softmax
        output = TimeDistributed(Dense(self.num_calltypes, activation='softmax'))(drop_3)
        
        # build model
        model = Model(inp, output)
        
        return model
    


    
    def build_fast_forked_masked_rnn(self):
        
        N, time, freqN = self.x_train.shape
        # inp_aud = Input(shape=(self.x_train[0].shape[1], self.x_train[0].shape[2], self.x_train[0].shape[3]))
        # inp_mask = Input(shape=(self.x_train[1].shape[1],))
        inp_aud = Input(shape=(None,freqN))
        
        # Add dummy channel dimension expected for Conv2D layers
        inp_chan = expand_dims(inp_aud, axis=-1)
               
        # Convolutional layers (conv - maxpool x3 )
        neighborhood = (3, 3)  # convolutional mask size
        c_1 = Conv2D(self.filters, neighborhood, padding='same', activation='relu')(inp_chan) #(inp_aud)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, neighborhood, padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, neighborhood, padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)       
        
        # Reshape   
        
        # frequency information axis has been reduced by max pooling
        # Last CNN output is time X freq_info X self.filters
        # Stack the filters:  time X (freq_info * self.filters)
        # Time steps will depend on the example or batch, so we use -1
        cnn_out = Reshape((-1, np.prod(mp_3.shape[-2:])))(mp_3)
        # reshape_1 = Reshape((self.x_train[0].shape[-3], -1))(mp_3)  
        
        # Determine which time slices should be ignored by computing a binary mask that
        # will be passed directly to the recurrent layers as a mask argument.
        mask = Masking()
        mask_val = mask.compute_mask(inp_aud)
        
        # bidirectional gated recurrent unit layers
        # Initial implementation had reucrrent_dropout=self.dropout for these layers,
        # but this prevents the use of the GRU implementation in the cuDNN library
        # which is an order of magnitude faster.
        gru_1 = GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,return_sequences=True)
        rnn_1 = Bidirectional(gru_1, merge_mode='mul')(cnn_out, mask=mask_val)
        # rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
        #                           recurrent_dropout=self.dropout,return_sequences=True), merge_mode='mul')(reshape_1, mask = inp_mask)#mask_tensor)
        gru_2 = GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,return_sequences=True)
        rnn_2 = Bidirectional(gru_2, merge_mode='mul')(rnn_1, mask=mask_val)
        # rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
        #                           recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)
        
        # 3x relu
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        
        # split into two to get two outputs
        dense_3A = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3A = Dropout(rate=self.dropout)(dense_3A)
        dense_3B = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3B = Dropout(rate=self.dropout)(dense_3B)
        
        # Fork into two outputs
        output_calltype = TimeDistributed(Dense(self.num_calltypes, activation='sigmoid'), name="calltype")(drop_3A)
        output_callpresence = TimeDistributed(Dense(2, activation='softmax'), name="callpresence")(drop_3B)
        
        # build model
        model = Model([inp_aud], [output_calltype, output_callpresence])
        
        return model
    
    
    
    
    
    def build_forked_masked_rnn(self):
        
        inp_aud = Input(shape=(self.x_train[0].shape[1], self.x_train[0].shape[2], self.x_train[0].shape[3]))
        inp_mask = Input(shape=(self.x_train[1].shape[1],))
                

        # Convolutional layers (conv - maxpool x3 )
        c_1 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(inp_aud) # would be (mask)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)
        
        
        # reshape
        reshape_1 = Reshape((self.x_train[0].shape[-3], -1))(mp_3)
        
        
        # bidirectional gated recurrent unit x2
        # mask goes here
        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  #recurrent_dropout=self.dropout,
                                  return_sequences=True), merge_mode='mul')(reshape_1, mask = inp_mask)#mask_tensor)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  #recurrent_dropout=self.dropout, 
                                  return_sequences=True), merge_mode='mul')(rnn_1, mask = inp_mask)
        
        # 3x relu
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        
        # split into two to get two outputs
        dense_3A = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3A = Dropout(rate=self.dropout)(dense_3A)
        dense_3B = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3B = Dropout(rate=self.dropout)(dense_3B)
        
        # Fork into two outputs
        output_calltype = TimeDistributed(Dense(self.num_calltypes, activation='sigmoid'), name="output_calltype")(drop_3A)
        output_callpresence = TimeDistributed(Dense(2, activation='softmax'), name="output_callpresence")(drop_3B)

        
        # build model
        model = Model([inp_aud, inp_mask], [output_calltype, output_callpresence])
        
        return model
    
    def build_masked_rnn(self):
        
        inp_aud = Input(shape=(self.x_train[0].shape[1], self.x_train[0].shape[2], self.x_train[0].shape[3]))
        inp_mask = Input(shape=(self.x_train[1].shape[1],))
                

        # Convolutional layers (conv - maxpool x3 )
        c_1 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(inp_aud) # would be (mask)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)
        
        
        # reshape
        reshape_1 = Reshape((self.x_train[0].shape[-3], -1))(mp_3)
                
        # bidirectional gated recurrent unit x2
        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout,return_sequences=True), merge_mode='mul')(reshape_1, mask = inp_mask)#mask_tensor)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)
        
        # 3x dense layers
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=self.dropout)(dense_3)
        
        # Output
        output_calltype = TimeDistributed(Dense(self.num_calltypes, activation='softmax'), name="output_calltype")(drop_3)

        # build model
        model = Model([inp_aud, inp_mask], [output_calltype])
                
        return model
    
    def build_forked_rnn(self):
        
        inp = Input(shape=(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3]))
        
        # Convolutional layers (conv - maxpool x3 )
        c_1 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(inp)# would be (mask)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)
        
        # reshape
        reshape_1 = Reshape((self.x_train.shape[-3], -1))(mp_3)
        
        # bidirectional gated recurrent unit x2
        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(reshape_1)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)
        
        # 3x relu
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        
        # split into two to get two outputs
        dense_3A = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3A = Dropout(rate=self.dropout)(dense_3A)
        dense_3B = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3B = Dropout(rate=self.dropout)(dense_3B)
        
        # Fork into two outputs
        output_calltype = TimeDistributed(Dense(self.num_calltypes, activation='sigmoid'), name="output_calltype")(drop_3A)
        output_callpresence = TimeDistributed(Dense(2, activation='softmax'), name="output_callpresence")(drop_3B)
        # hardcoded 2 here for call/notcall
        
        # build model
        model = Model(inp, [output_calltype, output_callpresence])
        
        return model
    
    
    def build_rnn_audio_Zacc(self):
                
        x_train_aud = self.x_train[0]
        x_train_acc_ch0 = self.x_train[1]
        inp_aud = Input(shape=(x_train_aud.shape[1], x_train_aud.shape[2], x_train_aud.shape[3]))
        inp_acc_0 = Input(shape=(x_train_acc_ch0.shape[1], x_train_acc_ch0.shape[2], 1))

        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_aud)
        aud = MaxPooling2D(pool_size=(1, 5))(aud)
        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Model(inputs=inp_aud, outputs=aud)

        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 5))(acc_0)
        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 2))(acc_0)
        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 2))(acc_0)
        acc_0 = Model(inputs=inp_acc_0, outputs=acc_0)


        combined = concatenate([aud.output, acc_0.output])#, acc_1.output, acc_2.output])
        combined = Reshape((x_train_aud.shape[-3], -1))(combined)

        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(combined)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)

        dense_1 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=self.dropout)(dense_3)

        output_aud = TimeDistributed(Dense(self.num_calltypes, activation='sigmoid'), name="output_aud")(drop_3)
        model = Model(inputs=[aud.input, acc_0.input], outputs=[output_aud])
        
        return model
    
    def build_rnn_audio_triacc(self):
        
        x_train_aud = x_train[0]
        x_train_acc_ch0 = x_train[1]
        x_train_acc_ch1 = x_train[2]
        x_train_acc_ch2 = x_train[3]
        inp_aud = Input(shape=(x_train_aud.shape[1], x_train_aud.shape[2], 1))
        inp_acc_0 = Input(shape=(x_train_acc_ch0.shape[1], x_train_acc_ch0.shape[2], 1))
        inp_acc_1 = Input(shape=(x_train_acc_ch1.shape[1], x_train_acc_ch1.shape[2], 1))
        inp_acc_2 = Input(shape=(x_train_acc_ch2.shape[1], x_train_acc_ch2.shape[2], 1))

        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_aud)
        aud = MaxPooling2D(pool_size=(1, 5))(aud)
        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(aud)
        aud = MaxPooling2D(pool_size=(1, 2))(aud)
        aud = Model(inputs=inp_aud, outputs=aud)

        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 5))(acc_0)
        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 2))(acc_0)
        acc_0 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_0)
        acc_0 = MaxPooling2D(pool_size=(1, 2))(acc_0)
        acc_0 = Model(inputs=inp_acc_0, outputs=acc_0)

        acc_1 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_acc_1)
        acc_1 = MaxPooling2D(pool_size=(1, 5))(acc_1)
        acc_1 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_1)
        acc_1 = MaxPooling2D(pool_size=(1, 2))(acc_1)
        acc_1 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_1)
        acc_1 = MaxPooling2D(pool_size=(1, 2))(acc_1)
        acc_1 = Model(inputs=inp_acc_1, outputs=acc_1)

        acc_2 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(inp_acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 5))(acc_2)
        acc_2 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        acc_2 = Conv2D(self.filters, (3, 3), padding='same', activation='relu')(acc_2)
        acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        acc_2 = Model(inputs=inp_acc_2, outputs=acc_2)

        combined = concatenate([aud.output, acc_0.output, acc_1.output, acc_2.output])

        combined = Reshape((x_train_aud.shape[-3], -1))(combined)

        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(combined)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout,
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)

        dense_1 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=self.dropout)(dense_3)

        output_aud = TimeDistributed(Dense(self.num_calltypes, activation='sigmoid'), name="output_aud")(drop_3)
        # output_foctype = Dense(3, activation='softmax', name="output_foctype")(drop_3)
        model = Model(inputs=[aud.input, acc_0.input, acc_1.input, acc_2.input], outputs=[output_aud])
        
        return model

        
        
    
    
