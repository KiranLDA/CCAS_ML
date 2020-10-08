#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:59:24 2020

@author: kiran
"""

# /home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess

import sys
# sys.path.append("/home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess/")
# # import my_module


import preprocess_functions

import ntpath
import re
import os
from glob import glob
from itertools import chain, compress
from random import random
#----------------------------------------------------------------------------------
# Hyena parameters
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
# Meerkat parameters
#----------------------------------------------------------------------------------



# paths
# audio_path = "/media/kiran/Kiran Meerkat/meerkat_detector/data/raw_data/AUDIO2/HM_LT_R07_20170821-20170825/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092.wav"
# label_path = "/home/kiran/Documents/ML/labels_CSV/labels_CSV/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092_label.csv"
spec_window_size = 1
slide = 0.5

# for spectrogram generation
# i = 3646 #3697#3683#3681#0#5334#
# start = i
# stop = i + spec_window_size
fft_win = 0.03
fft_hop = fft_win/8
n_mels = 128

# for label munging
sep='\t'
engine = None
start_column = "Start"
duration_column = "Duration"
label_column = "Name"
convert_to_seconds = True
label_for_other = "oth"
label_for_noise = "noise"
label_for_startstop = ['start', 'stop', 'skip', 'end']
call_types = {
    'cc' :["cc","Marker", "Marque"],
    'sn' :["sn","subm", "short","^s$", "s "],#"\\b[s]" "^((?!s).)*$" "s ", "+s", "s+", 
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL", "so "],
    'al' :["al","ALARM"],
    'beep':["beep"],
    'synch':["sync"],
    # 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
    # 'ukn':["ukn","unknown","UKN","UNKNOWN"]
    # 'nf' :["nf","nonfoc","NONFOC"],
    # 'noise':["x","X"]
    # 'overlap':"%"
    'oth':["oth","other","lc", "lost",
           "hyb","HYBRID","fu","sq","\+",
           "ukn","unknown",
           # "nf","nonfoc",
           "x",
           "\%","\*","\#","\?","\$"
           ],
    'noise':['start','stop','end','skip']
    }

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

# Must delete later
# label_filenames = [label_filenames[i] for i in [5,10,15,55,60]]

#Start the loop by going over every single labelled file id
for file_ID in label_filenames:
    # file_ID = label_filenames[2]
        
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, label_for_other, engine)
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
    
    #save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"

    # Don't run the code if that file has already been processed
    if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
        continue
    # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
    label_table.to_csv(os.path.join(save_label_table_path, save_label_table_filename), header=True, index=None, sep=' ', mode='a')
    
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
            
            spectro = generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                           n_mels = n_mels, window='hann', 
                                           fft_win= fft_win, fft_hop = fft_hop)
               
            #generate the label matrix
            label_matrix = create_label_matrix(label_table, spectro, call_types, start, 
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

            np.save(os.path.join(save_spec_path, save_spec_filename), spectro)     
            np.save(os.path.join(save_mat_path, save_mat_filename), label_matrix) 
            # np.save(os.path.join(save_spec_path, save_both_filename), np.array((spectro, label_matrix))) 


print(skipped_files)



# # code for plotting
# import matplotlib.pyplot as plt
# import matplotlib
# fig = plt.figure(figsize=(8, 4))
# a = fig.add_subplot(1,2,1)
# my_cmap = matplotlib.cm.get_cmap('magma')
# a.set_title('Spectrogram')
# librosa.display.specshow(test[0,:,:,0],cmap=my_cmap, y_axis='mel', x_axis='time')

# b = fig.add_subplot(1,2,2)
# plt.xticks(np.arange(0, label_matrix.shape[1], label_matrix.shape[1]/6), 
#             np.arange(start, stop, ((stop-start)/6)))
# plt.yticks(np.arange(0, len(label_matrix.index), 1), label_matrix.index)
# b.set_title('Actual')
# plt.xlabel('Time')
# b.imshow(test[0,:,:,0], aspect='30')
# plt.tight_layout()



###############################################################################################
# train = np.load("/home/kiran/Documents/animal_data_tmp/hyena/rmishra/dataset/x_train.npy")
# test = np.load("/home/kiran/Documents/animal_data_tmp/hyena/rmishra/dataset/x_test.npy")
# val = np.load("/home/kiran/Documents/animal_data_tmp/hyena/rmishra/dataset/x_val.npy")
# #dimensions are : number of spectrograms, time dimension, mel bands and number of channels


#----------------------------------------------------------------------------------
# Setup training and test sets
#----------------------------------------------------------------------------------

# Do it randomly for now
import os
from random import shuffle
from math import floor

# decide which files to shuffle
# def get_file_list_from_dir(datadir):
# all_files = os.listdir(os.path.abspath(datadir))
# data_files = list(filter(lambda file: file.endswith('.npy'), all_files))

# shuffle the audio data and use 75% for training and 15% for testing
file_list = label_filenames
shuffle(file_list)

split = 0.75
split_index = floor(len(file_list) * split)
training = file_list[:split_index]
testing = file_list[split_index:]


# Get the spectrogram dataset
spec_npy_filelist = os.listdir(save_spec_path)
x_train = []
for training_file in training:
    for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
        spectrogrammi = np.load(save_spec_path + '/' + spectro_npy)
        x_train.append(spectrogrammi.T)

# x_train = np.asarray(x_train)
np.save(os.path.join(train_test_path , 'x_train.npy'), x_train)
  

  
x_test = []
for training_file in testing:
    for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
        spectrogrammi = np.load(save_spec_path + '/' + spectro_npy)
        x_test.append(spectrogrammi.T)

# x_test = np.asarray(x_test)
np.save(os.path.join(train_test_path , 'x_test.npy'), x_test)
 
# Get the matching label dataset
spec_npy_filelist = os.listdir(save_mat_path)
y_train = []
for training_file in training:
    for spectro_npy in [s for s in spec_npy_filelist if training_file in s]:
        spectrogrammi = np.load(save_mat_path + '/' + spectro_npy)
        y_train.append(spectrogrammi.T)

# y_train = np.asarray(y_train)
np.save(os.path.join(train_test_path , 'y_train.npy'), y_train)
    
y_test = []
for training_file in testing:
    for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
        spectrogrammi = np.load(save_mat_path + '/' + spectro_npy)
        y_test.append(spectrogrammi.T)

# y_test = np.asarray(y_test)
np.save(os.path.join(train_test_path , 'y_test.npy'), y_test)
 
    
 
# # code for plotting
# import matplotlib.pyplot as plt
# import matplotlib
# fig = plt.figure(figsize=(8, 4))
# a = fig.add_subplot(1,2,1)
# my_cmap = matplotlib.cm.get_cmap('magma')
# a.set_title('Spectrogram')
# librosa.display.specshow(x_test[1,:,:,0],cmap=my_cmap, y_axis='mel', x_axis='time')

# b = fig.add_subplot(1,2,2)
# plt.yticks(np.arange(0, y_test[1,:,:,0].shape[0], y_test[1,:,:,0].shape[0]/6), 
#             np.arange(0, y_test[1,:,:,0].shape[0], y_test[1,:,:,0].shape[0]/6))
# plt.xticks(np.arange(0, y_test[1,:,:,0].shape[1], 1), y_test[1,:,:,0].shape[1])
# b.set_title('Actual')
# plt.xlabel('Time')
# b.imshow(y_test[1,:,:,0], aspect='30')
# plt.tight_layout()