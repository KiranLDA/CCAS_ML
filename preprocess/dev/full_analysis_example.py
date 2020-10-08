#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:37:12 2020

@author: kiran
"""

# /home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess

import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module


# from preprocess_functions import *
import preprocess.preprocess_functions as pre

import numpy as np
import librosa
import warnings
import ntpath
import re
import os
from glob import glob
from itertools import chain, compress
from random import random


#----------------------------------------------------------------------------------
# Hyena parameters - for xml later
#----------------------------------------------------------------------------------

# # paths
# audio_path = "/home/kiran/Documents/animal_data_tmp/hyena/rmishra/cc16_ML/cc16_352a/cc16_352a_14401s_audio.wav"
# label_path = "/home/kiran/Documents/animal_data_tmp/hyena/rmishra/cc16_ML/cc16_352a/cc16_352a_14401s_calls.txt"
# spec_window_size = 6 #mk 1
# slide = 3 #mk0.75


# # for spectrogram generation
# i = 3434#260
# start = i
# stop = i + spec_window_size
# fft_win = 0.047
# fft_hop = fft_win/2
# n_mels = 64 

# # for label munging
# sep='\s*\t\s*'
# engine = 'python'
# start_column = "StartTime"
# duration_column = "Duration"
# label_column = "Label"
# convert_to_seconds = False
# label_for_other = "OTH"
# label_for_noise = "NOISE"
# call_types = {    
#     'GIG':["gig"],
#     'SQL':["sql"],
#     'GRL':["grl"],
#     'GRN':["grn", "moo"],
#     'SQT':["sqt"],
#     # 'MOO':["moo"],
#     'RUM':["rum"],
#     'WHP':["whp"],
#     'OTH':["oth"]
#     }




#----------------------------------------------------------------------------------
# Meerkat parameters for xml - later
#----------------------------------------------------------------------------------

#------------------
# File paths
#------------------
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

#basically the root directory for saving spectrograms and labels
save_data_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'



# Note that the lines below don't need to be modified 
# unless you have a different file structure
# They will create a specific file sub directory pattern in save_data_path

# where to save the label tables
save_label_table_path = os.path.join(save_data_path, 'label_table')

# where to save the training specs and labels
train_path = os.path.join(save_data_path,'train_data')
save_spec_train_path = os.path.join(train_path + "spectrograms")
save_mat_train_path = os.path.join(train_path + "label_matrix")

# where to save the training labels and specs
test_path = os.path.join(save_data_path, 'test_data')
save_spec_test_path = os.path.join(test_path + "spectrograms")
save_mat_test_path = os.path.join(test_path + "label_matrix")


#------------------
# rolling window parameters
spec_window_size = 1
slide = 0.5

#------------------
# fast fourier parameters for mel spectrogram generation
fft_win = 0.01 #0.03
fft_hop = fft_win/2
n_mels = 30 #128

#------------------
# label munging parameters i.e. reading in audition or raven files
sep='\t'
engine = None
start_column = "Start"
duration_column = "Duration"
label_column = "Name"
convert_to_seconds = True
label_for_other = "oth"
label_for_noise = "noise"
label_for_startstop = ['start', 'stop', 'skip', 'end']


#------------------
# call dictionary - 
# this is a dictionary containing as keys the category you want your ML algo to output
# and for each call category, how it is likely to be noted in the label column of the audition or raven file
# For example, Marker is usually for a close call.
# Note that these are regural expressions and are not case sensitive
call_types = {
    'cc' :["cc","Marker", "Marque"],
    'sn' :["sn","subm", "short","^s$", "s "],
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL", "so "],
    'al' :["al","ALARM"],
    'beep':["beep"],
    'synch':["sync"],
    'oth':["oth","other","lc", "lost",
           "hyb","HYBRID","fu","sq","\+",
           "ukn","unknown",          
           "x",
           "\%","\*","\#","\?","\$"
           ],
    'noise':['start','stop','end','skip']
    }


# parameters that might be useful later that currently aren't dealt with
# we might want to make a separate category e.g. for focal and non-focal
# 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
# 'ukn':["ukn","unknown","UKN","UNKNOWN"]
# 'nf' :["nf","nonfoc","NONFOC"],
# 'noise':["x","X"]
# 'overlap':"%"
# 'nf':["nf","nonfoc"],'[]



#----------------------------------------------------------------------------------
# Actual bit running the code
#----------------------------------------------------------------------------------

# if any files are skipped because they are problematic, they are put here 
skipped_files =[]


# create the directories for saving the files 
#-----------------------------------------------------------------
for diri in [train_path, test_path , save_label_table_path]:
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

# Must delete later
# label_filenames = [label_filenames[i] for i in [5,10,15,55,60]]

#----------------------------------------------------------------------------------------------------
# Split the label filenames into training and test files
#----------------------------------------------------------------------------------------------------


from random import shuffle
from math import floor
# split = 0.75
# file_list = label_filenames
# shuffle(file_list)

# # randomly divide the files into those in the training, validation and test datasets
# split_index = floor(len(file_list) * split)
# training_filenames = file_list[:split_index]
# testing_filenames = file_list[split_index:]

# subset for testing purposes
# training_filenames = [training_filenames[i] for i in [1,5,10,15,55]]
# testing_filenames = [testing_filenames[i] for i in [5,2]]


#------------------------------------------------------------------
# because the training and testing filenames are random for each run, 
# lets keep a track record of the ones that were used
#------------------------------------------------------------------
# with open(os.path.join(save_label_table_path, "training_files_used.txt"), "w") as f:
#     for s in training_filenames:
#         f.write(str(s) +"\n")


# with open(os.path.join(save_label_table_path, "testing_files_used.txt"), "w") as f:
#     for s in testing_filenames:
#         f.write(str(s) +"\n")


# load the saved file
with open(os.path.join(save_label_table_path, "training_files_used.txt")) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
training_filenames = [x.strip() for x in content] 

with open(os.path.join(save_label_table_path, "testing_files_used.txt")) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
testing_filenames = [x.strip() for x in content] 


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       TRAINING
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(save_spec_train_path):
    os.mkdir(save_spec_train_path)
if not os.path.exists(save_mat_train_path):
    os.mkdir(save_mat_train_path)


# Start the loop by going over every single labelled file id
for file_ID in training_filenames:
    # file_ID = label_filenames[2]
        
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    
    # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    
    print ("File being processed : " + label_path)
    
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, 
                                   label_for_other, l abel_for_noise, engine, True)
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
    
    #save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"

    # Don't run the code if that file has already been processed
    # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
    #     continue
    # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
    label_table.to_csv(os.path.join(save_label_table_path, save_label_table_filename), header=True, index=None, sep=';')
    
    # load the audio data
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # # Reshaping the Audio file (mono) to deal with all wav files similarly
    # if y.ndim == 1:
    #     y = y.reshape(1, -1)
    
    # # Implement this for acc data
    # for ch in range(y.shape[0]):
    # ch=0
    # y_sub = y[:,ch]
    y_sub = y
    
    for rowi in range(len(label_table["Start"])):
        # loop through every labelled call event in the dataset and create a random start time to generate sprectrogram window
        random_start = label_table["Start"][rowi] - (random() * spec_window_size)
        
        #don't generate a label for the start stop skipon skipoff
        if label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False)[rowi]:
            continue
        
        # generate 3 spectrograms for every labelled call 
        #(idea is that it should include either 2 call + 1 noise or 1 call and 2 noise - balances the dataset a little)
        for start in np.arange(random_start, (random_start + (3*slide) ), slide):
            stop = start + spec_window_size
            
            
            if stop > label_table["End"][len(label_table["End"])-1]:
                continue
            if start < label_table["Start"][0]:
                continue
            
            spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                           n_mels = n_mels, window='hann', 
                                           fft_win= fft_win, fft_hop = fft_hop, normalise=True)
               
            #generate the label matrix
            label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                       stop, label_for_other, label_for_noise)
            
            
            # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
            file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
            if len(file_label) > 1 and 'noise' in file_label:
                file_label.remove('noise')
            category = '_'.join(file_label)
            
            # Save these files
            save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            # save_both_filename = file_ID + "_BOTH_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            
            np.save(os.path.join(save_spec_train_path, save_spec_filename), spectro)     
            np.save(os.path.join(save_mat_train_path, save_mat_filename), label_matrix) 
            # np.save(os.path.join(save_spec_path, save_both_filename), np.array((spectro, label_matrix))) 


print(skipped_files)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       TESTING
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
 
if not os.path.exists(test_path):
    os.mkdir(test_path)
if not os.path.exists(save_spec_test_path):
    os.mkdir(save_spec_test_path)
if not os.path.exists(save_mat_test_path):
    os.mkdir(save_mat_test_path)


#Start the loop by going over every single labelled file id
for file_ID in testing_filenames:
    # file_ID = label_filenames[2]
        
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    
    #if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    print ("File being processed : " + label_path)
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, 
                                   convert_to_seconds, label_for_other, label_for_noise, engine, True)
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
 
    # find the start and stop  of the labelling periods (also using skipon/skipoff)
    loop_table = label_table.loc[label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False), ["Label","Start"]]
    loop_times = list(loop_table["Start"])
    
    # Make sure that the file contains the right number of start and stops, otherwise go to the next file
    if len(loop_times)%2 != 0:
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
        skipped_files.append(label_path)    
        continue 
    
    #save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"

    # Don't run the code if that file has already been processed
    # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
    #     continue
    # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
    label_table.to_csv(os.path.join(save_label_table_path, save_label_table_filename), header=True, index=None, sep=';')
    
    
    # load the audio data
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    
    # # Reshaping the Audio file (mono) to deal with all wav files similarly
    # if y.ndim == 1:
    #     y = y.reshape(1, -1)
    
    # # Implement this for acc data
    # for ch in range(y.shape[0]):
    # ch=0
    # y_sub = y[:,ch]
    y_sub = y
    
    
    # loop through every labelling start based on skipon/off within this loop_table
    for loopi in range(0, int(len(loop_times)), 2): 
        #testing: loopi = 0
        fromi = loop_times[loopi]
        toi = loop_times[int(loopi + 1)] #define the end of the labelling periods
    
        for spectro_slide in np.arange(fromi, toi, slide):
            # spectro_slide = fromi
            # print(spectro_slide)
            start = spectro_slide
            stop = spectro_slide + spec_window_size
            
            #ignore cases where the window is larger than what is labelled (e.g. at the end)
            if stop <= toi:
                # generate the spectrogram
                spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                           n_mels = n_mels, window='hann', 
                                           fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                
                #generate the label matrix
                label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                           stop, label_for_other, label_for_noise)
                
                
                # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                if len(file_label) > 1 and 'noise' in file_label:
                    file_label.remove('noise')
                category = '_'.join(file_label)
                
                # Save these files
                save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                # save_both_filename = file_ID + "_BOTH_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                
                # save_spec_path = os.path.join(test_path + "spectrograms")
                # save_mat_path = os.path.join(test_path + "label_matrix")


                np.save(os.path.join(save_spec_test_path, save_spec_filename), spectro)     
                np.save(os.path.join(save_mat_test_path, save_mat_filename), label_matrix) 
                # np.save(os.path.join(save_spec_path, save_both_filename), np.array((spectro, label_matrix))) 



print(skipped_files)

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       CREATE THE TRAINING AND VALIDATION DATASETS FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
from random import shuffle
from math import floor

# save_spec_path = os.path.join(train_path + "spectrograms")
# save_mat_path = os.path.join(train_path + "label_matrix")

spectro_list = os.listdir(save_spec_train_path)
label_list = os.listdir(save_mat_train_path)


c = list(range(len(spectro_list)))
shuffle(c)
spectro_list = [spectro_list[i] for i in c]
label_list = ['_MAT_'.join(x.split("_SPEC_")) for x in spectro_list]


# randomly divide the files into those in the training, validation 
split = 0.75
split_index = floor(len(spectro_list) * split)

x_train_files = spectro_list[:split_index]
y_train_files = label_list[:split_index]
x_val_files = spectro_list[split_index:]
y_val_files = label_list[split_index:]

#join full path
x_train_filelist = [os.path.join(save_spec_train_path, x) for x in x_train_files]
y_train_filelist = [os.path.join(save_mat_train_path, x) for x in y_train_files]
x_val_filelist = [os.path.join(save_spec_train_path, x) for x in x_val_files]
y_val_filelist = [os.path.join(save_mat_train_path, x) for x in y_val_files]

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       BATCH GENERATOR FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer/")
import model.batch_generator as bg

batch = 32
epochs = 16#16

train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, True)
val_generator = bg.Batch_Generator(x_val_filelist, y_val_filelist, batch, True)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       Have a look at the batch
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

x_train, y_train = next(train_generator)
x_train.shape

# y_train.shape

# plot example spectrogram and label to make sure they match
# i = 29
# spec = x_train[i,:,:,0]
# label = y_train[i,:,:]

# plt.figure(figsize=(10, 12))
# plt.subplot(211)
# librosa.display.specshow(spec.T, x_axis='time' , y_axis='mel')
# plt.colorbar(format='%+2.0f dB')

# # plot LABEL
# plt.subplot(212)
# xaxis = range(0, np.flipud(label.T).shape[1]+1)
# yaxis = range(0, np.flipud(label.T).shape[0]+1)#
# plt.yticks(range(len(call_types.keys())),list(call_types.keys())[::-1])#
# plt.pcolormesh(xaxis, yaxis, np.flipud(label.T))
# plt.xlabel('time (s)')
# plt.ylabel('Calltype')
# plt.colorbar(label="Label")
# # plot_matrix(np.flipud(label), zunits='Label')

# plt.show()



#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       TRAIN RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------



import datetime

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D



# do it again so that the batch is different
# train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, True)

num_calltypes = y_train.shape[2]
filters = 128 #y_train.shape[1] #
gru_units = y_train.shape[1] #128
dense_neurons = 1024
dropout = 0.5


inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)

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


RNN_model.fit_generator(train_generator, 
                        steps_per_epoch = train_generator.steps_per_epoch(),
                        epochs = epochs,
                        callbacks = [early_stopping, reduce_lr_plat],
                        # shuffle = True,
                        validation_data = val_generator,
                        validation_steps = val_generator.steps_per_epoch() )



date_time = datetime.datetime.now()

date_now = str(date_time.date())

time_now = str(date_time.time())

sf = "/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_shuffle_test_" + date_now + "_" + time_now

if not os.path.isdir(sf):
        os.makedirs(sf)

RNN_model.save(sf + '/savedmodel' + '.h5')




#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       VIEW PREDICTIONS
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# save_spec_path = os.path.join(test_path + "spectrograms")
# save_mat_path = os.path.join(test_path + "label_matrix")


#-----------------------------------------------------
#load the model
# RNN_model = keras.models.load_model(sf + '/savedmodel' + '.h5')
# first nice working model is :
     # '/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_shuffle_test_2020-06-26_08:22:34.551302/savedmodel.h5'
#-----------------------------------------------------
# create a list of all the files
x_test_files = os.listdir(save_spec_test_path)
y_test_files = os.listdir(save_mat_test_path) 

#append full path to test files
x_test_files = [os.path.join(save_spec_test_path, x) for x in x_test_files ]
y_test_files = [os.path.join(save_mat_test_path, x) for x in y_test_files ]

#-----------------------------------------------------
# Predict over a batch

# #only need one epoch anyway
# batch = 64

# #get the test files into the data generator
# test_generator = Test_Batch_Generator(x_test_files , y_test_files, batch)

# #predict
# predictions = RNN_model.predict_generator(test_generator, test_generator.steps_per_epoch()+1 )


#-----------------------------------------------------
# Predict for a radom file
#-----------------------------------------------------
import postprocess.visualise_prediction_functions as pp

# meerkat calltype options
# ['_cc_' ,'_sn_' ,'_mo_' ,'_agg_','_ld_' ,'_soc_','_al_' ,'_beep_','_synch_','_oth_', '_noise_']

#   Find a random spectrogram of a particular call type
spec, label, pred = pp.predict_label_from_random_spectrogram(RNN_model, "_sn_", save_spec_test_path, save_mat_test_path)

# and plot it
pp.plot_spec_label_pred(spec[0,:,:,0], label, pred[0,:,:], list(call_types.keys())[::-1])
 
