#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:37:12 2020

@author: kiran
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          IMPORT LIBRARIES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# add path to local functions
# import sys
import os
os.chdir("/home/kiran/Documents/github/CCAS_ML")
# sys.path.append("/home/kiran/Documents/github/CCAS_ML")

# import own functions
import preprocess.preprocess_functions as pre
import postprocess.evaluation_metrics_functions as metrics
import postprocess.merge_predictions_functions as ppm
import model.batch_generator as bg
import postprocess.visualise_prediction_functions as pp
from model.callback_functions import LossHistory


# 
import numpy as np
import librosa
import warnings
import ntpath
# import re
import os
# from glob import glob
from itertools import compress  #chain, 
from random import random, shuffle
from math import floor
import statistics
import glob


# ML section packages
import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import TimeDistributed, Dense, Dropout, BatchNormalization
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
from keras.callbacks import TensorBoard

# postprocessfrom decimal import Decimal
from decimal import Decimal
import pandas as pd
import pickle

# from params import *

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS - will likely put them in another directory
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

other_ignored_in_training = True
run_name = "NoiseAugmented_ProportionallyWeighted_NoOther"

#------------------
# File paths
#------------------
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

# basically the root directory for train, test and model
save_data_path = os.path.join('/media/kiran/D0-P1/animal_data/meerkat', run_name)
if not os.path.isdir(save_data_path):
        os.makedirs(save_data_path)

#####
# Note that the lines below don't need to be modified 
# unless you have a different file structure
# They will create a specific file sub directory pattern in save_data_path
#####


# because we are still in a development phase, I want to use the same training and testing sets for each model

# Training folders
train_path = os.path.join(save_data_path,'train_data')
if not os.path.isdir(train_path):
        os.makedirs(train_path)

# train_path = "/media/kiran/D0-P1/animal_data/meerkat/NoiseAugmented_NoOther/train_data"
        
save_spec_train_path = os.path.join(train_path , "spectrograms")
if not os.path.isdir(save_spec_train_path):
        os.makedirs(save_spec_train_path)
        
save_mat_train_path = os.path.join(train_path , "label_matrix")
if not os.path.isdir(save_mat_train_path):
        os.makedirs(save_mat_train_path)
        
save_label_table_train_path = os.path.join(train_path, 'label_table')
if not os.path.isdir(save_label_table_train_path):
        os.makedirs(save_label_table_train_path)


# Test folders
test_path = os.path.join(save_data_path, 'test_data')
if not os.path.isdir(test_path):
        os.makedirs(test_path)
        
save_spec_test_path = os.path.join(test_path , "spectrograms")
if not os.path.isdir(save_spec_test_path):
        os.makedirs(save_spec_test_path)
        
save_mat_test_path = os.path.join(test_path , "label_matrix")
if not os.path.isdir(save_mat_test_path):
        os.makedirs(save_mat_test_path)

save_pred_test_path = os.path.join(test_path , "predictions")
if not os.path.isdir(save_pred_test_path):
        os.makedirs(save_pred_test_path)
        
        
save_metrics_path = os.path.join(test_path , "metrics")
if not os.path.isdir(save_metrics_path):
        os.makedirs(save_metrics_path)
        
save_pred_stack_test_path = os.path.join(save_pred_test_path,"stacks")
if not os.path.isdir(save_pred_stack_test_path):
        os.makedirs(save_pred_stack_test_path)

save_pred_table_test_path = os.path.join(save_pred_test_path,"pred_table")
if not os.path.isdir(save_pred_table_test_path):
        os.makedirs(save_pred_table_test_path)
        
save_label_table_test_path = os.path.join(test_path, 'label_table')
if not os.path.isdir(save_label_table_test_path):
        os.makedirs(save_label_table_test_path)


# Model folder
save_model_path = os.path.join(save_data_path, 'trained_model')
if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
        
save_tensorboard_path = os.path.join(save_model_path, 'tensorboard_logs')
if not os.path.isdir(save_tensorboard_path):
    os.makedirs(save_tensorboard_path)      

#------------------
# rolling window parameters
spec_window_size = 1
slide = 0.5

#------------------
# fast fourier parameters for mel spectrogram generation
fft_win = 0.01 #0.03
fft_hop = fft_win/2
n_mels = 30 #128
window = "hann"
normalise = True

#------------------
# ML parameters
dense_neurons = 1024
dropout = 0.5
filters = 128 #y_train.shape[1] #

#------------------
# split between the training and the test set
train_test_split = 0.90
train_val_split = 0.75

#------------------
# data augmentation parameters
n_steps = -2 # for pitch shift
stretch_factor = 0.99 #If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
scaling_factor = 0.1
random_range = 0.1
#-----------------------------
# thresholding parameters
low_thr = 0.2

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
# 'overlap':["%"]
# 'nf':["nf","nonfoc"]

#------------------
#### ML parameters
batch = 32
epochs = 100#16 #16



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#             1 - PREPROCESSING - SETTING UP DIRECTORIES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


# if any files are skipped because they are problematic, they are put here 
skipped_files =[]



# Find the input data
#-----------------------------------------------------------------

# find all label paths
EXT = "*.csv"
label_filepaths = []
for PATH in label_dirs:
     label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])


# find all audio paths (will be longer than label path as not everything is labelled)
audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
     audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])
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
#----------------------------------------------------------------------------------------------------
#         1.1. Split the label filenames into training and test files
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# If the training and testing files exists then load them, otherwise create them
if os.path.exists(os.path.join(save_model_path, "training_files_used.txt")):
    # load the saved file
    with open(os.path.join(save_model_path, "training_files_used.txt")) as f:
        content = f.readlines()    
    training_filenames = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line
    with open(os.path.join(save_model_path, "testing_files_used.txt")) as f:
        content = f.readlines()    
    testing_filenames = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line
# otherwiss create the training and testing files
else: 
    # randomise the order of the files
    file_list = label_filenames
    shuffle(file_list)
    
    # randomly divide the files into those in the training and test datasets
    split_index = floor(len(file_list) * train_test_split)
    training_filenames = file_list[:split_index]
    testing_filenames = file_list[split_index:]

    # save a copy of the training and testing diles
    with open(os.path.join(save_model_path, "training_files_used.txt"), "w") as f:
        for s in training_filenames:
            f.write(str(s) +"\n")
    with open(os.path.join(save_model_path, "testing_files_used.txt"), "w") as f:
        for s in testing_filenames:
            f.write(str(s) +"\n")




#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       2 - TRAINING
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------------------------------
# 2.1. Generate and save the training files
#----------------------------------------------------------------------------------------------------



# Start the loop by going over every single labelled file id
for file_ID in training_filenames:
    # file_ID = label_filenames[2]
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
    
    # only generate training files if they don't already exist    
    if not os.path.exists(os.path.join(save_label_table_train_path, save_label_table_filename)):
            
        # find the matching audio for the label data
        audio_path = [s for s in audio_filepaths if file_ID in s][0]
        
        # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
        label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
        
        print ("File being processed : " + label_path)    
        
        # create a standardised table which contains all the labels of that file - also can be used for validation
        label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, 
                                       label_for_other, label_for_noise, engine, True)
        # replace duration of beeps with 0.04 seconds - meerkat particularity
        label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
        label_table.loc[label_table["beep"] == True, "End"] += 0.04
    
        # Don't run the code if that file has already been processed
        # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
        #     continue
        # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
        label_table.to_csv(os.path.join(save_label_table_train_path, save_label_table_filename), header=True, index=None, sep=';')
        
        #save the label tables with other, but for the purpose of labelling, remove other
        if other_ignored_in_training:
            label_table = label_table[label_table[label_for_other] == False]
            label_table= label_table.reset_index(drop=True)
        
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
            #rowi=0
            # loop through every labelled call event in the dataset and create a random start time to generate sprectrogram window
            random_start = label_table["Start"][rowi] - (random() * spec_window_size)
            
            #don't generate a label for the start stop skipon skipoff
            if label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False)[rowi]:
                continue
            
            # generate 3 spectrograms for every labelled call 
            #(idea is that it should include either 2 call + 1 noise or 1 call and 2 noise - balances the dataset a little)
            for start in np.arange(random_start, (random_start + (3*slide) ), slide):
                # start = random_start
                stop = start + spec_window_size
                
                
                if stop > label_table["End"][len(label_table["End"])-1]:
                    continue
                if start < label_table["Start"][0]:
                    continue
                
                spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                               n_mels = n_mels, window=window, 
                                               fft_win= fft_win, fft_hop = fft_hop, normalise=normalise)
                   
                #generate the label matrix
                label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                           stop, label_for_noise)
                
                
                # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                if len(file_label) > 1 and label_for_noise in file_label:
                    file_label.remove(label_for_noise)
                category = '_'.join(file_label)
                
                # Save these files
                save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                # save_both_filename = file_ID + "_BOTH_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
                
                np.save(os.path.join(save_spec_train_path, save_spec_filename), spectro)     
                np.save(os.path.join(save_mat_train_path, save_mat_filename), label_matrix) 
                # np.save(os.path.join(save_spec_path, save_both_filename), np.array((spectro, label_matrix))) 


# save the files that were skipped
print(skipped_files)


# save a copy of the training and testing diles
with open(os.path.join(save_model_path, "skipped_training_files.txt"), "w") as f:
    for s in skipped_files:
        f.write(str(s) +"\n")


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#  2.2. Data augmentation
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# save_label_table_path = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/label_table"

calls = list(call_types.keys())
files = os.listdir(save_mat_train_path)

# count the calls in the training dataset
calls_number = dict()
for calli in calls:
    calls_number[calli] = len([i for i in files if calli in i] )

# find how many more or less calls there are than the mean (to see how much imbalance there is)
calls_needed = calls_number.copy()
mean_call = statistics.mean(calls_number[k] for k in calls_number)
for i in calls_needed:
    calls_needed[i] = int(mean_call - calls_number[i]) #save_pred_table_test_path = os.path.join(save_pred_test_path,"pred_table")
if not os.path.isdir(save_pred_table_test_path):
        os.makedirs(save_pred_table_test_path)

# how many times would the dataset need to be augmented to reach the average number of calls
calls_augmented = calls_needed.copy()
for i in calls_augmented:
    calls_augmented[i] = calls_needed[i]/(calls_number[i]+0.001)


# find the ones which might need to be augmentedsave_pred_table_test_path = os.path.join(save_pred_test_path,"pred_table")
if not os.path.isdir(save_pred_table_test_path):
        os.makedirs(save_pred_table_test_path)
calls_to_augment = ["_"+ k for k, v in calls_augmented.items() if v > 0]

# don't augment other or noise
if "_"+ label_for_other in calls_to_augment:
    calls_to_augment.remove("_"+ label_for_other)
if "_"+ label_for_noise in calls_to_augment:
    calls_to_augment.remove("_"+ label_for_noise)


# find all the files
all_training = [os.path.join(save_spec_train_path, x) for x in os.listdir(save_spec_train_path)] #[x for x in os.listdir(spectro_folder)]
for calltype in calls_to_augment:   
    # calltype = calls_to_augment[0]
    spec_filepaths =  glob.glob(save_spec_train_path+'/*'+calltype+'*') # find all files of that calltype
    noise_filepaths = glob.glob(save_spec_train_path+'/*' + label_for_noise + '*')     
    for spec_filepath in spec_filepaths: 
        # spec_filepath = spec_filepaths[0]
        # noise aug
        spec_filepath= [spec_filepath]
        aug_data, aug_spectro, aug_mat, aug_spec_filename, aug_mat_filename = pre.augment_with_noise(spec_filepath, noise_filepaths, 
                                                                                                             audio_filepaths,calltype, scaling_factor, 
                                                                                                             other_ignored_in_training,
                                                                                                             random_range, spec_window_size,
                                                                                                             n_mels, window, fft_win, fft_hop,
                                                                                                             normalise, save_label_table_train_path,
                                                                                                             call_types,label_for_other,label_for_noise)
        # save these spectrograms
        np.save(os.path.join(save_spec_train_path, aug_spec_filename), aug_spectro)     
        np.save(os.path.join(save_mat_train_path, aug_mat_filename), aug_mat) 

        # # pitch shift
        # augmented_data, augmented_spectrogram, augmented_label, spec_name, label_name = pre.augment_with_pitch_shift(spec_filepaths, audio_filepaths, calltype, n_steps,other_ignored_in_training,
        #                                                                                 random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
        #                                                                                 save_label_table_path, call_types, label_for_noise)
        
        
        # # time shft
        
        # augmented_data, augmented_spectrogram, augmented_label, spec_name, label_name = pre.augment_with_time_stretch(spec_filepaths, audio_filepaths, calltype, stretch_factor,other_ignored_in_training,
        #                                                                                 random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
        #                                                                                 save_label_table_path, call_types, label_for_noise)



#----------------------------------------------------------------------------------------------------
#   2.3. CREATE THE TRAINING AND VALIDATION DATASETS FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------


# save_spec_path = os.path.join(train_path + "spectrograms")
# save_mat_path = os.path.join(train_path + "label_matrix")

spectro_list = os.listdir(save_spec_train_path)
label_list = os.listdir(save_mat_train_path)

#randomise the list of labels
c = list(range(len(spectro_list)))
shuffle(c)
spectro_list = [spectro_list[i] for i in c]
label_list = ['_MAT_'.join(x.split("_SPEC_")) for x in spectro_list]

# randomly divide the files into those in the training, validation based on split
split_index = floor(len(spectro_list) * train_val_split)
x_train_files = spectro_list[:split_index]
y_train_files = label_list[:split_index]
x_val_files = spectro_list[split_index:]
y_val_files = label_list[split_index:]

#join full path
x_train_filelist = [os.path.join(save_spec_train_path, x) for x in x_train_files]
y_train_filelist = [os.path.join(save_mat_train_path, x) for x in y_train_files]
x_val_filelist = [os.path.join(save_spec_train_path, x) for x in x_val_files]
y_val_filelist = [os.path.join(save_mat_train_path, x) for x in y_val_files]


# save a copy of the training and val files
with open(os.path.join(save_model_path,  "training_specs.txt"), "w") as f:
    for s in x_train_filelist:
        f.write(str(s) +"\n")

with open(os.path.join(save_model_path,  "training_mats.txt"), "w") as f:
    for s in y_train_filelist:
        f.write(str(s) +"\n")

# save a copy of the training and testing diles
with open(os.path.join(save_model_path, "validation_specs.txt"), "w") as f:
    for s in x_val_filelist:
        f.write(str(s) +"\n")
        
with open(os.path.join(save_model_path, "validation_mats.txt"), "w") as f:
    for s in y_val_filelist:
        f.write(str(s) +"\n")


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#        2.4.  Weigh the calls according to the numbers available
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

calls = list(call_types.keys())
files = os.listdir(save_mat_train_path)

call_count = dict()
for calli in calls:
    call_count[calli] = len([i for i in files if calli in i] )
    

# find max and fivide max by all of these values
#give greatest weight to classes that are rarely seen
max_call = max([(value, key) for key, value in call_count.items()])[0]
weight_dict = call_count
weight_dict[label_for_other] = 1
for i in weight_dict:
    # weight_dict[i] = float(1-(weight_dict[i]/max_call))
    weight_dict[i] = float(max_call/weight_dict[i])

if other_ignored_in_training:
    weight_dict[label_for_other] = 0

train_weights_list = []
for i in x_train_filelist: 
    for k, v in weight_dict.items():
        if "_" + k + ".npy" in i:
            train_weights_list.append(v)

val_weights_list = []
for i in x_val_filelist: 
    for k, v in weight_dict.items():
        if "_" + k + ".npy" in i:
            val_weights_list.append(v)

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       2.5. BATCH GENERATOR FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer/")


train_generator = bg.Weighted_Batch_Generator(x_train_filelist, y_train_filelist, train_weights_list, batch, True)
val_generator = bg.Weighted_Batch_Generator(x_val_filelist, y_val_filelist, val_weights_list, batch, True)

# train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, True)
# val_generator = bg.Batch_Generator(x_val_filelist, y_val_filelist, batch, True)


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       Have a look at the batch
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

x_train, y_train, weight_train = next(train_generator)
# x_train, y_train = next(train_generator)
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
#       3. CONSTRUCT THE RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# do it again so that the batch is different
# train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, True)


# initial parameters
num_calltypes = y_train.shape[2]
gru_units = y_train.shape[1] 


#--------------------------------------------
# Construct the RNN
#--------------------------------------------


# Input
inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# Convolutional layers (conv - maxpool x3 )
c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)


# reshape
reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

# bidirectional gated recurrent unit x2
rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)

# 3x relu
dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
drop_1 = Dropout(rate=dropout)(dense_1)
dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
drop_2 = Dropout(rate=dropout)(dense_2)
dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
drop_3 = Dropout(rate=dropout)(dense_3)

# softmax
output = TimeDistributed(Dense(num_calltypes, activation='softmax'))(drop_3)

# build model
RNN_model = Model(inp, output)

# Adam optimiser
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
RNN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Setup callbycks: learning rate / loss /tensorboard
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
loss = LossHistory()

#tensorboard
tensorboard = TensorBoard(# Write to logs directory, e.g. logs/30Oct-05:00
                          log_dir = save_tensorboard_path, #"/media/kiran/D0-P1/animal_data/meerkat/NoiseAugmented_NoOther/trained_model/tensorboard_logs", #"logs/{}".format(time.strftime('%d%b-%H%M')),        
                          histogram_freq=0,
                          write_graph=True,  # Show the network
                          write_grads=True   # Show gradients
                          )    


# fit model
RNN_model.fit_generator(train_generator, 
                        steps_per_epoch = train_generator.steps_per_epoch(),
                        epochs = epochs,
                        callbacks = [early_stopping, reduce_lr_plat, loss, tensorboard],
                        validation_data = val_generator,
                        validation_steps = val_generator.steps_per_epoch())


# save the model
date_time = datetime.datetime.now()
date_now = str(date_time.date())
time_now = str(date_time.time())

sf = os.path.join(save_model_path, run_name+ "_" + date_now + "_" + time_now)
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


# # meerkat calltype options
# # ['_cc_' ,'_sn_' ,'_mo_' ,'_agg_','_ld_' ,'_soc_','_al_' ,'_beep_','_synch_','_oth_', '_noise_']

# #   Find a random spectrogram of a particular call type
# spec, label, pred = pp.predict_label_from_random_spectrogram(RNN_model, "_sn_", save_spec_test_path, save_mat_test_path)

# # and plot it
# pp.plot_spec_label_pred(spec[0,:,:,0], label, pred[0,:,:], list(call_types.keys())[::-1])
 




#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#               LOOP AND PREDICT OVER TEST FILES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

skipped_files = []
# testing_filenames = testing_filenames[7:]
#Start the loop by going over every single labelled file id
for file_ID in testing_filenames:
    # file_ID = testing_filenames[2]
    
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    
    #if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    
    print("*****************************************************************")   
    print("*****************************************************************") 
    print ("File being processed : " + label_path)
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, 
                                   convert_to_seconds, label_for_other, label_for_noise, engine, True)
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
    label_table.loc[label_table["beep"] == True, "End"] += 0.04
    
    # find the start and stop  of the labelling periods (also using skipon/skipoff)
    loop_table = label_table.loc[label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False), ["Label","Start"]]
    loop_times = list(loop_table["Start"])
    
    # Make sure that the file contains the right number of start and stops, otherwise go to the next file
    if len(loop_times)%2 != 0:
        print("!!!!!!!!!!!!!!!!")
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
        skipped_files.append(file_ID)
        # break
        continue 
    if len(loop_times) == 0:
        print("!!!!!!!!!!!!!!!!")
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
        skipped_files.append(file_ID)
        # break
        continue 
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
    
    # Don't run the code if that file has already been processed
    # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
    #     continue
    # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
    label_table.to_csv(os.path.join(save_label_table_test_path, save_label_table_filename), 
                       header=True, index=None, sep=';')
    
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
    
    # probabilities = []
    # for low_thr in [0.2]:
    # loop through every labelling start based on skipon/off within this loop_table
    for loopi in range(0, int(len(loop_times)), 2):
        # loopi = 0
        fromi =  loop_times[loopi]
        #toi = fromi + 5
        toi = loop_times[int(loopi + 1)] # define the end of the labelling periods
        
        # if the file exists, load it
        if os.path.exists(os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy')):
            pred_list = np.load( os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'))
        # if not, generate it
        else:
        
            pred_list = []
    
            for spectro_slide in np.arange(fromi, toi, slide):
                
                
                
                # spectro_slide = fromi
                start = round(spectro_slide,3)
                stop = round(spectro_slide + spec_window_size, 3)
                
                # start = round(start + slide, 3)
                # stop = round(spectro_slide + spec_window_size, 3)
                # ignore cases where the window is larger than what is labelled (e.g. at the end)
                if stop <= toi:
                    
                    # # Generate the relevant spectrogram name
                    # save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                    # save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                    # save_pred_filename = file_ID + "_PRED_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                    
                    spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                           n_mels = n_mels, window='hann', 
                                                           fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                    
                    label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                                           stop, label_for_noise)
                    
                    # Load the spectrogram
                    spec = spectro.T
                    spec = spec[np.newaxis, ..., np.newaxis]  
                    
                    # generate the prediction
                    pred = RNN_model.predict(spec)
                    
                    # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                    # file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                    # if len(file_label) > 1 and 'noise' in file_label:
                    #     file_label.remove('noise')
                    # category = '_'.join(file_label)
                    
                    # save_spec_filename = save_spec_filename + category + ".npy"
                    # save_mat_filename = save_mat_filename + category + ".npy"
                    # save_pred_filename = save_pred_filename + category + ".npy"
                    
                    # add this prediction to the stack that will be used to generate the predictions table
                    pred_list.append(np.squeeze(pred))
                    
            # save the prediction list  
            np.save( os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), pred_list)
            with open(os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                for row in pred_list:
                    f.write(str(row) +"\n")
                
        for low_thr in [0.2]:#[0.1,0.3]:
            for high_thr in [0.5,0.7,0.9]: #[0.5,0.7,0.8,0.9,0.95]: 

                
                low_thr = round(low_thr,2)                               
                high_thr = round(high_thr,2)

                save_pred_table_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt"
                
                #----------------------------------------------------------------------------
                # Compile the predictions for each on/off labelling chunk
                detections = ppm.merge_p(probabilities = pred_list, 
                                         labels=list(call_types.keys()),
                                         starttime = 0, 
                                         frameadv_s = fft_hop, 
                                         specadv_s = slide,
                                         low_thr=low_thr, 
                                         high_thr=high_thr, 
                                         debug=1)
                
                if len(detections) == 0:  
                    detections = pd.DataFrame(columns = ['category', 'start', 'end', 'scores'])
                
                pred_table = pd.DataFrame() 
                
                #convert these detections to a predictions table                
                table = pd.DataFrame(detections)
                table["Label"] = table["category"]
                table["Start"] = round(table["start"]*fft_hop + fromi, 3) #table["start"].apply(Decimal)*Decimal(fft_hop) + Decimal(fromi)
                table["Duration"] = round( (table["end"]-table["start"])*fft_hop, 3) #(table["end"].apply(Decimal)-table["start"].apply(Decimal))*Decimal(fft_hop)
                table["End"] = round(table["end"]*fft_hop + fromi, 3) #table["Start"].apply(Decimal) + table["Duration"].apply(Decimal)
                
                # keep only the useful columns    
                table = table[["Label","Start","Duration", "End", "scores"]]  
                
                # Add a row which stores the start of the labelling period
                row_start = pd.DataFrame()
                row_start.loc[0,'Label'] = list(loop_table["Label"])[loopi]
                row_start.loc[0,'Start'] = fromi
                row_start.loc[0,'Duration'] = 0
                row_start.loc[0,'End'] = fromi 
                row_start.loc[0,'scores'] = None
                
                # Add a row which stores the end of the labelling period
                row_stop = pd.DataFrame()
                row_stop.loc[0,'Label'] = list(loop_table["Label"])[int(loopi + 1)]
                row_stop.loc[0,'Start'] = toi
                row_stop.loc[0,'Duration'] = 0
                row_stop.loc[0,'End'] = toi 
                row_start.loc[0,'scores'] = None
                
                # put these rows to the label table
                table = pd.concat([row_start, table, row_stop]) 
                
                # add the true false columns based on the call types dictionary
                for true_label in call_types:
                    table[true_label] = False
                    for old_label in call_types[true_label]:
                        table.loc[table["Label"].str.contains(old_label, regex=True, case = False), true_label] = True
                
                # add this table to the overall predictions table for that collar
                pred_table = pd.concat([pred_table, table ])
                
                # for each on/off labelling chunk, we can save the prediction and append it to the previous chunk
                if loopi == 0:                    
                    # for the first chunck keep the header, but not when appending later
                    pred_table.to_csv(os.path.join(save_pred_table_test_path, save_pred_table_filename), 
                                      header=True, index=None, sep=';', mode = 'a')
                else:
                    pred_table.to_csv(os.path.join(save_pred_table_test_path, save_pred_table_filename), 
                                      header=None, index=None, sep=';', mode = 'a')
                
                
                
        
'''
# load the saved file
with open(os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK.txt')) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
pred_list = [x.strip() for x in content] 


#or
pred_list = np.load( os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK.npy'))
   
'''
        
# save the files that were skipped
print(skipped_files)

# save a copy of the training and testing diles
with open(os.path.join(save_model_path, "skipped_testing_files.txt"), "w") as f:
    for s in skipped_files:
        f.write(str(s) +"\n")
       
##############################################################################################
# Loop through tables and remove duplicates of rows (bevause files are created through appending)

pred_tables = glob.glob(save_pred_table_test_path+ "/*PRED_TABLE*.txt")
for file in pred_tables:
    df = pd.read_csv(file, delimiter=';') 
    # df = df.drop_duplicates(keep=False)
    df = df.loc[df['Label'] != 'Label']
    df.to_csv(file, header=True, index=None, sep=';', mode = 'w')


##############################################################################################
#
#    EVALUATE
#
##############################################################################################

#########################################################################
##  Create overall thresholds
#########################################################################

# skipped = [os.path.split(path)[1] for path in skipped_files]
file_ID_list = [file_ID for file_ID in testing_filenames if file_ID not in skipped_files]
label_list =  [os.path.join(save_label_table_test_path,file_ID + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]
for low_thr in [0.2]:#[0.1,0.3]:
    for high_thr in [0.5,0.7,0.9]: #[0.5,0.7,0.8,0.9,0.95]: 
# for low_thr in [0.1,0.3]:
#     for high_thr in [0.5,0.7,0.8,0.9,0.95]: 
# for low_thr in [0.1]:
#     for high_thr in [0.2,0.3,0.4]: 
        
        low_thr = round(low_thr,2)                               
        high_thr = round(high_thr,2) 
        
        pred_list = [os.path.join(save_pred_table_test_path,file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt" ) for file_ID in file_ID_list ]
        evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
        Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
        
        # specify file names
        precision_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Precision.csv'
        recall_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Recall.csv'
        cat_frag_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Category_fragmentation.csv'
        time_frag_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Time_fragmentation.csv'
        confusion_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv'
        gt_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Label_indices.csv"
        pred_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Prection_indices.csv"
        match_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Matching_table.txt"
        timediff_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Time_difference.txt"    
        
        # save files
        Prec.to_csv( os.path.join(save_metrics_path, precision_filename))
        Rec.to_csv( os.path.join(save_metrics_path, recall_filename))
        cat_frag.to_csv( os.path.join(save_metrics_path, cat_frag_filename))
        time_frag.to_csv(os.path.join(save_metrics_path, time_frag_filename))
        cf.to_csv(os.path.join(save_metrics_path, confusion_filename))
        gt_indices.to_csv(os.path.join(save_metrics_path, gt_filename ))
        pred_indices.to_csv(os.path.join(save_metrics_path, pred_filename ))                  
        with open(os.path.join(save_metrics_path, match_filename), "wb") as fp:   #Picklin
                  pickle.dump(match, fp)
        with open(os.path.join(save_metrics_path, timediff_filename), "wb") as fp:   #Pickling
            pickle.dump(offset, fp)    


#########################################################################
# plot overall confusion matrix
#########################################################################

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

# normalise = True
# for low_thr in [0.1,0.3]:
#     for high_thr in [0.5,0.7,0.8,0.9,0.95]: 
for low_thr in [0.2]: # [0.1,0.3]:
    for high_thr in [0.5,0.7,0.9]: # [0.5,0.7,0.8,0.9,0.95]: 
        
        low_thr = round(low_thr,2)                               
        high_thr = round(high_thr,2) 
        confusion_filename = os.path.join(save_metrics_path, "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv')
        with open(confusion_filename, newline='') as csvfile:
            array = list(csv.reader(csvfile))
    
        df_cm = pd.DataFrame(array)#, range(6), range(6))    
        
        #get rid of the weird indentations and make rows and columns as names
        new_col = df_cm.iloc[0] #grab the first row for the header
        df_cm = df_cm[1:] #take the data less the header row
        df_cm.columns = new_col #set the header row as the df header    
        new_row = df_cm['']
        df_cm = df_cm.drop('', 1)
        df_cm.index = new_row
        df_cm.index.name= None
        df_cm.columns.name= None
        
        # # replace FP and FN with noise
        df_cm['noise'] = df_cm['FN'] 
        df_cm.loc['noise']=df_cm.loc['FP']
        
        # remove FP and FN
        df_cm = df_cm.drop("FN", axis=1)
        df_cm = df_cm.drop("FP", axis=0)
        ####        
        
        df_cm = df_cm.apply(pd.to_numeric)
        # #move last negatives to end
        # col_name = "FN"
        # last_col = df_cm.pop(col_name)
        # df_cm.insert(df_cm.shape[1], col_name, last_col)
        
        # # remove noi        for low_thr in [0.1,0.3]:
            # for high_thr in [0.5,0.7,0.8,0.9,0.95]: 
        
        #normalise the confusion matrix
        if normalise == True:
            # divide_by = df_cm.sum(axis=1)
            # divide_by.index = new_header
            # new_row = df_cm.index 
            # new_col = df_cm.columns
            df_cm = df_cm.div(df_cm.sum(axis=1), axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
            # df_cm.index = new_row
            # df_cm.columns = new_col
        
        # plt.figure(figsize=(10,7))
        ax = plt.axes()
        sn.set(font_scale=1.1) # for label size
        sn.heatmap((df_cm), annot=True, annot_kws={"size": 10}, ax= ax) # font size
        ax.set_title(str(low_thr) + "-" + str(high_thr) )
        plt.savefig(os.path.join(save_metrics_path, "Confusion_mat_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
        plt.show()




# #########################################################################
# ## Create thresholds per file
# #########################################################################

# # loop over each test file
# for file_ID in file_ID_list: #file_ID = testing_filenames[0]
#     label_ID = file_ID + "_LABEL_TABLE.txt"
#     label_list = [os.path.join(save_label_table_test_path,label_ID)]
#     #loop over each threshold
#     for low_thr in [0.1,0.3]:
#         for high_thr in [0.5,0.7,0.9]: 
            
#             low_thr = round(low_thr,1)                               
#             high_thr = round(high_thr,1)       

#             pred_ID = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt"
                    
#             pred_list = [os.path.join(save_pred_table_test_path, pred_ID)]
#             evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
#             Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
            
#             # specify file names
#             precision_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Precision.csv'
#             recall_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Recall.csv'
#             cat_frag_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Category_fragmentation.csv'
#             time_frag_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Time_fragmentation.csv'
#             confusion_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv'
#             gt_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Label_indices.csv"
#             pred_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Prection_indices.csv"
#             match_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Matching_table.txt"
#             timediff_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Time_difference.txt"
            
#             # save files
#             Prec.to_csv( os.path.join(save_metrics_path, precision_filename))
#             Rec.to_csv( os.path.join(save_metrics_path, recall_filename))
#             cat_frag.to_csv( os.path.join(save_metrics_path, cat_frag_filename))
#             time_frag.to_csv(os.path.join(save_metrics_path, time_frag_filename))
#             cf.to_csv(os.path.join(save_metrics_path, confusion_filename))
#             gt_indices.to_csv(os.path.join(save_metrics_path, gt_filename ))
#             pred_indices.to_csv(os.path.join(save_metrics_path, pred_filename ))
#             with open(os.path.join(save_metrics_path, match_filename), "wb") as fp:   #Picklin
#                       pickle.dump(match, fp)
#             with open(os.path.join(save_metrics_path, timediff_filename), "wb") as fp:   #Pickling
#                 pickle.dump(offset, fp)    


