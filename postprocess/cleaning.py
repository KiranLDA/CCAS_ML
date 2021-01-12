#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:03:50 2020

@author: kiran
"""
import pandas as pd
import glob
import os
from itertools import compress
import ntpath

def find_audio_and_label_files(label_dirs, audio_dirs, to_rm):
    
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
    for name in to_rm:
        audio_filepaths = list(compress(audio_filepaths, [name not in filei for filei in audio_filepaths]))
    
    
    # Find the names of the recordings
    audio_filenames = [os.path.splitext(ntpath.basename(wavi))[0] for wavi in audio_filepaths]
    label_filenames = []
    for filepathi in label_filepaths:
        for audio_nami in audio_filenames:
            if audio_nami in filepathi:
                label_filenames.append(audio_nami)
                
    return label_filenames, label_filepaths, audio_filenames, audio_filepaths
    

def split_into_train_testsets(save_model_path, label_filenames, train_test_split):
    # If the training and testing files exists then load them, otherwise create them
    #----------
    
    if os.path.exists(os.path.join(save_model_path, "training_files_used.txt")):
        # load the saved file
        with open(os.path.join(save_model_path, "training_files_used.txt")) as f:
            content = f.readlines()    
        training_filenames = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line
        with open(os.path.join(save_model_path, "testing_files_used.txt")) as f:
            content = f.readlines()    
        testing_filenames = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line
    
    # otherwiss create the training and testing files
    #----------
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
    return training_filenames, testing_filenames



def compile_pred_table(detections, fft_hop, fromi, toi, loop_table, loopi, call_types):

    
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
    
    return pred_table
    

