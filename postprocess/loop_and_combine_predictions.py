#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Fri Jun 26 13:09:58 2020

@author: kiran
'''

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
from decimal import Decimal
import pandas as pd
import decimal


import postprocess.merge_predictions_functions as ppm

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
#----------------------------------------------------------------------------------
#           PARAMETER SETUP - Meerkat parameters to later go in xml
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# give this specific model a name
name_model_run = 'model_shuffle_test_2020-06-26_08:22:34.551302'
# directory where model is saved
save_model_path = '/media/kiran/D0-P1/animal_data/meerkat/saved_models/'


#------------------
# File paths
#------------------
# location of the raw data
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

# location of the processed data i.e. the root directory for saving spectrograms and labels
save_data_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'

# Note that the lines below don't need to be modified 
# unless you have/want a different file structure
# They will create a specific file sub directory pattern in save_data_path

# 1. where to save the label tables
# save_label_table_path = os.path.join(save_data_path, 'label_table')
# save_pred_table_path = os.path.join(save_data_path, 'prediction_table')

# 2. where to save the training specs and labels
train_path = os.path.join(save_data_path,'train_data')
save_spec_train_path = os.path.join(train_path , "spectrograms")
save_mat_train_path = os.path.join(train_path , "label_matrix")
save_label_table_train_path = os.path.join(train_path, 'label_table')


# 3. where to save the testing labels and specs and predictions
test_path = os.path.join(save_data_path, 'test_data')
save_spec_test_path = os.path.join(test_path , "spectrograms")
save_mat_test_path = os.path.join(test_path , "label_matrix")
save_pred_test_path = os.path.join(test_path , "predictions")
save_label_table_test_path = os.path.join(test_path, 'label_table')
save_pred_table_test_path = os.path.join(test_path, 'pred_table')


#if those directories do not exist, create them
directories = [save_data_path, 
               train_path, save_spec_train_path, save_mat_train_path, save_label_table_train_path,
               test_path, save_spec_test_path, save_mat_test_path, save_label_table_test_path, save_pred_test_path, save_pred_table_test_path]

for diri in directories:
    if not os.path.exists(diri):
        os.mkdir(diri)
        
        
#------------------
# rolling window parameters
spec_window_size = 1.
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
# call dictionary 
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



#--------------------
# Prediction double threshold parameters
low_thr = 0.4 
high_thr = 0.8


#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#                    DATA MUNGING
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# if any files are skipped because they are problematic, they are put here 
skipped_files =[]

# (done further up and therefore commented out)
# # create the directories for saving the files 
# #-----------------------------------------------------------------
# for diri in [train_path, test_path , save_label_table_path]:
#     if not os.path.exists(diri):
#         os.mkdir(diri)

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



# Find training and test data
#-----------------------------------------------------------------


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

# specify where to stort the information
train_log = os.path.join(save_model_path, name_model_run, "training_files_used.txt") 
test_log = os.path.join(save_model_path, name_model_run, "testing_files_used.txt")

# with open(train_log, "w") as f:
#     for s in training_filenames:
#         f.write(str(s) +"\n")

# with open(test_log , "w") as f:
#     for s in testing_filenames:
#         f.write(str(s) +"\n")

# load the saved file
with open(train_log) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
training_filenames = [x.strip() for x in content] 

with open(test_log) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
testing_filenames = [x.strip() for x in content] 

# testing_filenames = list(np.setdiff1d(label_filenames, training_filenames))

testing_filenames = testing_filenames[5:]

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#               LOAD IN THE TRAINED MODEL
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
from keras.models import load_model

RNN_model = load_model(os.path.join(save_model_path,name_model_run,'savedmodel.h5'))

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#               LOOP AND PREDICT OVER TEST FILES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

#Start the loop by going over every single labelled file id
for file_ID in testing_filenames:
    # file_ID = testing_filenames[1]
        
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
 
    # find the start and stop  of the labelling periods (also using skipon/skipoff)
    loop_table = label_table.loc[label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False), ["Label","Start"]]
    loop_times = list(loop_table["Start"])
    
    # Make sure that the file contains the right number of start and stops, otherwise go to the next file
    if len(loop_times)%2 != 0:
        print("!!!!!!!!!!!!!!!!")
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
        skipped_files.append(label_path)
        # break
        continue 
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
    save_pred_table_filename = file_ID + "_PRED_TABLE.txt"
    
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
    
    probabilities = []
    pred_table = pd.DataFrame()
    
    # loop through every labelling start based on skipon/off within this loop_table
    for loopi in range(0, int(len(loop_times)), 2): 
        # loopi = 0
        fromi =  loop_times[loopi]
        #toi = fromi + 5
        toi = loop_times[int(loopi + 1)] # define the end of the labelling periods
        
        pred_list = []
    
        for spectro_slide in np.arange(fromi, toi, slide):
            # spectro_slide = fromi
            start = round(spectro_slide,3)#round(Decimal(spectro_slide), 3)#
            stop = round(spectro_slide + spec_window_size, 3)# start + Decimal(spec_window_size)#
            
            # start = round(start + slide, 3)
            # stop = round(spectro_slide + spec_window_size, 3)
            #ignore cases where the window is larger than what is labelled (e.g. at the end)
            if stop <= toi:
                
                # Generate the relevant spectrogram name
                save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                save_pred_filename = file_ID + "_PRED_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                        
                # # if the spectrogram exists read it in
                # if any(save_spec_filename in x for x in os.listdir(save_spec_test_path)):
                #     spectro = np.load(os.path.join(save_spec_test_path, 
                #                                     os.listdir(save_spec_test_path)[np.min(np.where([save_spec_filename in x for x in os.listdir(save_spec_test_path)]))]))
                # else:
                #     # otherwise generate the spectrogram
                #     spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                #                                n_mels = n_mels, window='hann', 
                #                                fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                
                # # If the label matrix exists then read it in
                # if any(save_mat_filename in x for x in os.listdir(save_mat_test_path)):
                #     label_matrix = pd.DataFrame(np.load(os.path.join(save_mat_test_path, 
                #                                         os.listdir(save_mat_test_path)[np.min(np.where([save_mat_filename in x for x in os.listdir(save_mat_test_path)]))]),),
                #                                 columns = np.arange(start=start, stop=stop, step=(stop-start)/spectro.shape[1]),
                #                                 index = call_types.keys())
                # else:
                #     # generate the label matrix
                #     label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                #                                stop, label_for_other, label_for_noise)
                                    
                # # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                # file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                # if len(file_label) > 1 and 'noise' in file_label:
                #     file_label.remove('noise')
                # category = '_'.join(file_label)
                
                # save_spec_filename = save_spec_filename + category + ".npy"
                # save_mat_filename = save_mat_filename + category + ".npy"
                # save_pred_filename = save_pred_filename + category + ".npy"
                
                
                
                # If the prediction exists then read in all the files (since they have to exist to have generated the spectrogram)
                if any(save_pred_filename in x for x in os.listdir(save_pred_test_path)):
                    # if the spectrogram exists read it in
                    if any(save_spec_filename in x for x in os.listdir(save_spec_test_path)):
                        spectro = np.load(os.path.join(save_spec_test_path, 
                                                        os.listdir(save_spec_test_path)[np.min(np.where([save_spec_filename in x for x in os.listdir(save_spec_test_path)]))]))
                    else:
                        # otherwise generate the spectrogram
                        spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                   n_mels = n_mels, window='hann', 
                                                   fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                    # If the label matrix exists then read it in
                    if any(save_mat_filename in x for x in os.listdir(save_mat_test_path)):
                        label_matrix = pd.DataFrame(np.load(os.path.join(save_mat_test_path, 
                                                            os.listdir(save_mat_test_path)[np.min(np.where([save_mat_filename in x for x in os.listdir(save_mat_test_path)]))]),),
                                                    columns = np.arange(start=start, stop=stop, step=(stop-start)/spectro.shape[1]),
                                                    index = call_types.keys())
                    else:
                        # generate the label matrix
                        label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                                   stop, label_for_other, label_for_noise)
                    
                    #Get prediction
                    pred = pd.DataFrame(np.load(os.path.join(save_pred_test_path, 
                                                        os.listdir(save_pred_test_path)[np.min(np.where([save_pred_filename in x for x in os.listdir(save_pred_test_path)]))]),),
                                                columns = np.arange(start=start, stop=stop, step=(stop-start)/spectro.shape[1]),
                                                index = call_types.keys())
                    
                
                # otherwise, see if those matrices exist and generate if non existent
                else:
                    # if the spectrogram exists read it in
                    if any(save_spec_filename in x for x in os.listdir(save_spec_test_path)):
                        spectro = np.load(os.path.join(save_spec_test_path, 
                                                        os.listdir(save_spec_test_path)[np.min(np.where([save_spec_filename in x for x in os.listdir(save_spec_test_path)]))]))
                    else:
                        # otherwise generate the spectrogram
                        spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                   n_mels = n_mels, window='hann', 
                                                   fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                    # If the label matrix exists then read it in
                    if any(save_mat_filename in x for x in os.listdir(save_mat_test_path)):
                        label_matrix = pd.DataFrame(np.load(os.path.join(save_mat_test_path, 
                                                            os.listdir(save_mat_test_path)[np.min(np.where([save_mat_filename in x for x in os.listdir(save_mat_test_path)]))]),),
                                                    columns = np.arange(start=start, stop=stop, step=(stop-start)/spectro.shape[1]),
                                                    index = call_types.keys())
                    else:
                        # generate the label matrix
                        label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                                   stop, label_for_other, label_for_noise)
                    
                    # Load the spectrogram
                    spec = spectro.T
                    spec = spec[np.newaxis, ..., np.newaxis]  
                    
                    # generate the prediction
                    pred = RNN_model.predict(spec)
                 
                
                
                
                # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                if len(file_label) > 1 and 'noise' in file_label:
                    file_label.remove('noise')
                category = '_'.join(file_label)
                
                save_spec_filename = save_spec_filename + category + ".npy"
                save_mat_filename = save_mat_filename + category + ".npy"
                save_pred_filename = save_pred_filename + category + ".npy"
                                
                

                # add this prediction to the stack that will be used to generate the predictions table
                pred_list.append(np.squeeze(pred))
                
                # Find matching label file and load it
                # label = label_matrix.T
                
                # Save the files 
                if not os.path.isfile(os.path.join(save_spec_test_path, save_spec_filename)):
                    np.save(os.path.join(save_spec_test_path, save_spec_filename), spectro) 
                if not os.path.isfile(os.path.join(save_mat_test_path, save_mat_filename)):
                    np.save(os.path.join(save_mat_test_path, save_mat_filename), label_matrix)  
                if not os.path.isfile(os.path.join(save_pred_test_path, save_pred_filename)):
                    np.save(os.path.join(save_pred_test_path, save_mat_filename), pred.T) 
            
            
            
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
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

        #convert these detections to a predictions table                
        table = pd.DataFrame(detections)
        table["Label"] = table["category"]
        table["Start"] = table["start"].apply(Decimal)*Decimal(fft_hop) + Decimal(fromi) 
        table["Duration"] = (table["end"].apply(Decimal)-table["start"].apply(Decimal))*Decimal(fft_hop)
        table["End"] = table["Start"].apply(Decimal) + table["Duration"].apply(Decimal)
        
        #keep only the useful columns
        # table = table[["Label","Start","Duration", "End"]]            
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
    
    # once we have gone over all the on/off labelling chunks, we can save the predictions
    pred_table.to_csv(os.path.join(save_pred_table_test_path, save_pred_table_filename), header=True, index=None, sep=';')






print(skipped_files)


