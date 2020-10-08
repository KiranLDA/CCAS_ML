#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:25:19 2020

@author: kiran
"""

# /home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess

# import sys
# sys.path.append("/home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess/")
# # import my_module


# import preprocess_functions

import ntpath
# import re
import os
from glob import glob
from itertools import chain, compress


# #----------------------------------------------------------------------------------
# # Meerkat parameters
# #----------------------------------------------------------------------------------

# # paths
# # audio_path = "/media/kiran/Kiran Meerkat/meerkat_detector/data/raw_data/AUDIO2/HM_LT_R07_20170821-20170825/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092.wav"
# # label_path = "/home/kiran/Documents/ML/labels_CSV/labels_CSV/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092_label.csv"
# spec_window_size = 1
# slide = 0.5

# # for spectrogram generation
# # i = 3646 #3697#3683#3681#0#5334#
# # start = i
# # stop = i + spec_window_size
# fft_win = 0.03
# fft_hop = fft_win/8
# n_mels = 128

# # for label munging
# sep='\t'
# engine = None
# start_column = "Start"
# duration_column = "Duration"
# label_column = "Name"
# convert_to_seconds = True
# label_for_other = "oth"
# label_for_noise = "noise"
# label_for_startstop = ['start', 'stop', 'skip', 'end']
# call_types = {
#     'cc' :["cc","Marker", "Marque"],
#     'sn' :["sn","subm", "short","^s$", "s "],#"\\b[s]" "^((?!s).)*$" "s ", "+s", "s+", 
#     'mo' :["mo","MOV","MOVE"],
#     'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
#     'ld' :["ld","LD","lead","LEAD"],
#     'soc':["soc","SOCIAL", "so "],
#     'al' :["al","ALARM"],
#     'beep':["beep"],
#     'synch':["sync"],
#     # 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
#     # 'ukn':["ukn","unknown","UKN","UNKNOWN"]
#     # 'nf' :["nf","nonfoc","NONFOC"],
#     # 'noise':["x","X"]
#     # 'overlap':"%"
#     'oth':["oth","other","lc", "lost",
#            "hyb","HYBRID","fu","sq","\+",
#            "ukn","unknown",
#            # "nf","nonfoc",
#            "x",
#            "\%","\*","\#","\?","\$"
#            ],
#     'noise':['start','stop','end','skip']
#     }

# where to get the raw wav and labels
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]


# where to save the processed wavs and dirs ready for division into training and test sets
save_spec_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/spectrograms'
save_mat_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_matrix'
save_label_table_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_table'

train_test_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'

#----------------------------------------------------------------------------------
# Actual bit running the code
#----------------------------------------------------------------------------------

# if any files are skipped because they are problematic, they are put here 
skipped_files =[]


# create the directories for saving the files 
#-----------------------------------------------------------------
for diri in [save_spec_path, save_mat_path , save_label_table_path]:
    if not os.path.exists(diri):
        os.mkdir(diri)


# Find the input data
#-----------------------------------------------------------------

# find all label paths
EXT = "*.csv"
label_filepaths = []
for PATH in label_dirs:
     label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])


# find all audio paths (will be longer than label path as not everything is labelled)
audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
     audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])
# get rid of the focal follows (going around behind the meerkat with a microphone)
audio_filepaths = list(compress(audio_filepaths, ["SOUNDFOC" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["PROCESSED" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["LABEL" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["label" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["_SS" not in filei for filei in audio_filepaths]))


# Find the names of the recordings
audio_filenames = [os.path.splitext(ntpath.basename(wavi))[0] for wavi in audio_filepaths]
label_filenames = []
for filepathi in label_filepaths:
    for audio_nami in audio_filenames:
        if audio_nami in filepathi:
            label_filenames.append(audio_nami)


#----------------------------------------------------------------------------------
# Setup training and test sets for the data generator
#----------------------------------------------------------------------------------

# Do it randomly for now
import os
from random import shuffle
from math import floor
 
import train_test
# import importlib
# importlib.reload(train_test)

split_data = train_test.train_test(label_filenames,  0.75, save_spec_path, save_mat_path) 
# split the files randomly into training, validation, testing files  
x_train_filelist, y_train_filelist, x_val_filelist, y_val_filelist, x_test_filelist, y_test_filelist =  split_data.randomise_train_val_test()  

#----------------------------------------------------------------------------------
# put in data generator
#----------------------------------------------------------------------------------

from batch_generator import Batch_Generator
# import importlib
# importlib.reload(batch_generator)

#for xml later
batch = 24
epochs = 9 #16

# any more and it crashed

train_generator = Batch_Generator(x_train_filelist, y_train_filelist, batch)
val_generator = Batch_Generator(x_val_filelist, y_val_filelist, batch)
test_generator = Batch_Generator(x_test_filelist, y_test_filelist, batch)


#----------------------------------------------------------------------------------
# BUild model in tensorflow
#----------------------------------------------------------------------------------
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

#function parameters

x_train, y_train = next(train_generator)

num_calltypes = y_train.shape[2]
filters = 128 #y_train.shape[1] #
gru_units = y_train.shape[1]#128
dense_neurons = 1024
dropout=0.5




inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

# KD time delay recurrent nn - slower but could be replaced at a later stage
# GRU gaited recurrent network -feeds into future and past - go through each other
rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)

# KD goes back from flat to spectro
dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
drop_1 = Dropout(rate=dropout)(dense_1)
dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
drop_2 = Dropout(rate=dropout)(dense_2)
dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
drop_3 = Dropout(rate=dropout)(dense_3)

output = TimeDistributed(Dense(num_calltypes, activation='softmax'))(drop_3)
RNN_model = Model(inp, output)

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
RNN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)




# for data, labels in train_generator:
#     print(data.shape)



# training the model with generator
RNN_model.fit_generator(train_generator, 
                    steps_per_epoch = train_generator.steps_per_epoch(),
                    epochs = epochs,
                    callbacks = [early_stopping, reduce_lr_plat],
                    # class_weight = lossweight,
                    shuffle = True,
                    validation_data = val_generator,
                    validation_steps = val_generator.steps_per_epoch() )


date_time = datetime.datetime.now()
date_now = str(date_time.date())
time_now = str(date_time.time())
sf = "/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_test_" + date_now + "_" + time_now
if not os.path.isdir(sf):
        os.makedirs(sf)

RNN_model.save(sf + '/savedmodel' + '.h5')

# with open(sf + '/history.pickle', 'wb') as f:
#     pickle.dump(model_fit.history, f)


