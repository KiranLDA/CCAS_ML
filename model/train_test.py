#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:54:36 2020

@author: kiran
"""
    # Do it randomly for now
import os
from random import shuffle
from math import floor

class train_test():
    def __init__(self, label_filenames, split, save_spec_path, save_mat_path):
        self.label_filenames = label_filenames
        self.split = split
        self.save_spec_path = save_spec_path
        self.save_mat_path = save_mat_path

    
    def randomise_train_val_test(self):
        
        #------------------------------------------------------------------------------
        # shuffle the audio data and use 75% for training and 15% for testing
        file_list = self.label_filenames
        shuffle(file_list)
        
        #------------------------------------------------------------------------------
        # randomly divide the files into those in the training, validation and test datasets
        
        split_index = floor(len(file_list) * self.split)
        training = file_list[:split_index]
        validation = training[floor(len(training) * self.split):]
        training = training[:floor(len(training) * self.split)]
        testing = file_list[split_index:]
        
        
        #------------------------------------------------------------------------------
        ## TRAINING
        # Get the spectrogram dataset
        spec_npy_filelist = os.listdir(self.save_spec_path)
        x_train_filelist = []
        for training_file in training:
            for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
                x_train_filelist.append(self.save_spec_path + '/' + spectro_npy)
        
        
        # get the label dataset
        mat_npy_filelist = os.listdir(self.save_mat_path)
        y_train_filelist = []
        for training_file in training:
            for mat_npy in [s for s in mat_npy_filelist if training_file  in s]:
                y_train_filelist.append(self.save_mat_path + '/' + mat_npy)
        
        #------------------------------------------------------------------------------
        ## VALIDATION
        # Get the spectrogram dataset
        spec_npy_filelist = os.listdir(self.save_spec_path)
        x_val_filelist = []
        for training_file in validation:
            for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
                x_val_filelist.append(self.save_spec_path + '/' + spectro_npy)
        
        
        # get the label dataset
        mat_npy_filelist = os.listdir(self.save_mat_path)
        y_val_filelist = []
        for training_file in validation:
            for mat_npy in [s for s in mat_npy_filelist if training_file  in s]:
                y_val_filelist.append(self.save_mat_path + '/' + mat_npy)
                
                
        #------------------------------------------------------------------------------
        ## TESTING
        # Get the spectrogram dataset
        spec_npy_filelist = os.listdir(self.save_spec_path)
        x_test_filelist = []
        for training_file in testing:
            for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
                x_test_filelist.append(self.save_spec_path + '/' + spectro_npy)
        
        
        # get the label dataset
        mat_npy_filelist = os.listdir(self.save_mat_path)
        y_test_filelist = []
        for training_file in testing:
            for mat_npy in [s for s in mat_npy_filelist if training_file  in s]:
                y_test_filelist.append(self.save_mat_path + '/' + mat_npy)
        
        #return x_train_filelist, y_train_filelist, x_test_filelist, y_test_filelist

        return x_train_filelist, y_train_filelist, x_val_filelist, y_val_filelist, x_test_filelist, y_test_filelist
