#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:25:28 2020

@author: kiran

the aim of this code is going to be to:
    take an audio file: if spectrograms have been generated for that file pass
    if not generate spectrogram based on whether there is a label
    loop through all the spectrograms, predict and derive a label from the prediction
    put all the labels together
"""


# Build bacl label matrix
import os
import numpy as np
from itertools import groupby

test_mat_path = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/subset_4_testing/test_data/label_matrix"
test_spec_path = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/subset_4_testing/test_data/spectrograms"

#get all the test files
labels = os.listdir(test_mat_path)
specs = os.listdir(test_spec_path)

# sort the test files by recording/collar (all lavels from 1 recording go into the same list)
result = [list(group) for _, group in groupby(labels, key=lambda x: x.split("_MAT_")[0])]


# go over each collar/recording
for recording_labels in result:
    
    #store the id
    recording_ID = recording_labels[0].split("_MAT_")[0])
    
    # sort the list of recordings by starttime
    recording_labels.sort(key = lambda x: float(x.split("_MAT_")[1].split("s_")[0].split("s-")[0]))
    
    # # create an empty dataframe to store the data in
    overall_start = recording_labels[0].split("_MAT_")[1].split("s_")[0].split("s-")[0]
    overall_end = recording_labels[len(recording_labels)-1].split("_MAT_")[1].split("s_")[0].split("s-")[1]
          
    mat = np.load(os.path.join(test_mat_path,recording_labels[0]))    
    timesteps = mat.shape[1] # find number of columns for matrix
    colnames = np.arange(start=overall_start, stop=overall_stop, step=(overall_stop-overall_start)/timesteps)
    rownames = call_types.keys()

    
    #data might 
    big_mat = np.empty(shape=[len(call_types.keys()), 0])
    rownames = call_types.keys()
    
    
    # loop over overlapping files and complete
    for label_mat in recording_labels
        #get one spectrogram and the next
        recording, start, stop = extract_params_from_mat(label_mat)
        
             
        mini_mat = np.load(os.path.join(test_mat_path,label_mat))
        
        
        timesteps = mat.shape[1] # find number of columns for matrix
        colnames = np.arange(start=start, stop=stop, step=(stop-start)/timesteps)
        rownames = call_types.keys()

        
    

def extract_params_from_mat(file):
    recording_ID = file.split("_MAT_")[0]
    x = txt.split("_MAT_")[1]
    start_stop = x.split("s_")[0]
    start = float(start_stop.split("s-")[0])
    stop = float(start_stop.split("s-")[1])
    
    return recording_ID, start, stop

