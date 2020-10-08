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


class network():
    
    def __init__(self, x_train, num_calltypes, filters, gru_units, dense_neurons, dropout):
        self.x_train = x_train
        self.num_calltypes = num_calltypes
        self.filters = filters
        self.gru_units = gru_units
        self.dense_neurons = dense_neurons
        self.dropout = dropout
        
    def build_rnn_audio(self):
        
        inp = Input(shape=(self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3]))
        
        c_1 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(inp)
        mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
        c_2 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_1)
        mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
        c_3 = Conv2D(self.filters, (3,3), padding='same', activation='relu')(mp_2)
        mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)
        
        reshape_1 = Reshape((self.x_train.shape[-3], -1))(mp_3)
        
        # KD time delay recurrent nn - slower but could be replaced at a later stage
        # GRU gaited recurrent network -feeds into future and past - go through each other
        rnn_1 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(reshape_1)
        rnn_2 = Bidirectional(GRU(units=self.gru_units, activation='tanh', dropout=self.dropout, 
                                  recurrent_dropout=self.dropout, return_sequences=True), merge_mode='mul')(rnn_1)
        
        # KD goes back from flat to spectro
        dense_1  = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(rnn_2)
        drop_1 = Dropout(rate=self.dropout)(dense_1)
        dense_2 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_1)
        drop_2 = Dropout(rate=self.dropout)(dense_2)
        dense_3 = TimeDistributed(Dense(self.dense_neurons, activation='relu'))(drop_2)
        drop_3 = Dropout(rate=self.dropout)(dense_3)
        
        output = TimeDistributed(Dense(num_calltypes, activation='softmax'))(drop_3)
        model = Model(inp, output)
        
        return model
    
    
    def build_rnn_audio_Zacc(self):
        
        x_train_aud = x_train[0]
        x_train_acc_ch0 = x_train[1]
        # x_train_acc_ch1 = x_train[2]
        # x_train_acc_ch2 = x_train[3]
        inp_aud = Input(shape=(x_train_aud.shape[1], x_train_aud.shape[2], 1))
        inp_acc_0 = Input(shape=(x_train_acc_ch0.shape[1], x_train_acc_ch0.shape[2], 1))
        # inp_acc_1 = Input(shape=(x_train_acc_ch1.shape[1], x_train_acc_ch1.shape[2], 1))
        # inp_acc_2 = Input(shape=(x_train_acc_ch2.shape[1], x_train_acc_ch2.shape[2], 1))

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

        # acc_1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp_acc_1)
        # acc_1 = MaxPooling2D(pool_size=(1, 5))(acc_1)
        # acc_1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_1)
        # acc_1 = MaxPooling2D(pool_size=(1, 2))(acc_1)
        # acc_1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_1)
        # acc_1 = MaxPooling2D(pool_size=(1, 2))(acc_1)
        # acc_1 = Model(inputs=inp_acc_1, outputs=acc_1)

        # acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(inp_acc_2)
        # acc_2 = MaxPooling2D(pool_size=(1, 5))(acc_2)
        # acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_2)
        # acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        # acc_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(acc_2)
        # acc_2 = MaxPooling2D(pool_size=(1, 2))(acc_2)
        # acc_2 = Model(inputs=inp_acc_2, outputs=acc_2)

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

        output_aud = TimeDistributed(Dense(num_calltypes, activation='sigmoid'), name="output_aud")(drop_3)
        # output_foctype = Dense(3, activation='softmax', name="output_foctype")(drop_3)
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

        output_aud = TimeDistributed(Dense(num_calltypes, activation='sigmoid'), name="output_aud")(drop_3)
        # output_foctype = Dense(3, activation='softmax', name="output_foctype")(drop_3)
        model = Model(inputs=[aud.input, acc_0.input, acc_1.input, acc_2.input], outputs=[output_aud])
        
        return model

        
        
    
    
