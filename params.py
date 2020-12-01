#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:10:30 2020

@author: kiran
"""

import os

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
    'soc':["soc","SOCIAL", "so ", "so"],
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

