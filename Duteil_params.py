#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:27:13 2021
@author: kiran
"""

import os

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS - will likely put them in another directory
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

github_dir = "/home/mathieu/Documents/Git/CCAS_ML"
results_dir = '/media/mathieu/Elements/code/KiranLDA/results/'

min_scaling_factor = 0.3
max_scaling_factor = 0.8
run_name = "NEW_predation_NoiseAugmented_"+ str(min_scaling_factor)+"_" +str(max_scaling_factor)+"_NotWeighted_MaskedOther_Forked"
run_name = "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked"

#------------------
# File paths
#------------------
# label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
#             "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
label_dirs = ["/media/mathieu/Elements/data/Meerkat data/all_csv/predation_synched_call_tables"]


# audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
#              "/media/kiran/Kiran Meerkat/Meerkat data 2019"]
audio_dirs = ["/media/mathieu/Elements/data/Meerkat data/all_wav/recordings with labelled predation"]

acoustic_data_path = ["/media/mathieu/MPI_dirs/EAS_shared/meerkat/working/processed/acoustic/"]
acoustic_data_path = ["/media/mathieu/Elements/data/Meerkat data/all_csv"]

# basically the root directory for train, test and model
save_data_path = os.path.join(results_dir, run_name)
if not os.path.isdir(save_data_path):
        os.makedirs(save_data_path)

#####
# Note that the lines below don't need to be modified 
# unless you have a different file structure
# They will create a specific file sub directory pattern in save_data_path
#####


# Test folders
data = 'new_run'
test_path = os.path.join(save_data_path, data)
if not os.path.isdir(test_path):
        os.makedirs(test_path)
        

save_pred_test_path = os.path.join(test_path , "predictions")
if not os.path.isdir(save_pred_test_path):
        os.makedirs(save_pred_test_path)

save_pred_stack_test_path = os.path.join(save_pred_test_path,"stacks")
if not os.path.isdir(save_pred_stack_test_path):
        os.makedirs(save_pred_stack_test_path)        

save_pred_table_test_path = os.path.join(save_pred_test_path,"pred_table")
if not os.path.isdir(save_pred_table_test_path):
        os.makedirs(save_pred_table_test_path)
        
save_label_table_test_path = os.path.join(test_path, 'label_table')
if not os.path.isdir(save_label_table_test_path):
        os.makedirs(save_label_table_test_path)

save_metrics_path = os.path.join(test_path , "metrics")
if not os.path.isdir(save_metrics_path):
        os.makedirs(save_metrics_path)
        

# Model folder
save_model_path = os.path.join(save_data_path, 'trained_model')
if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
        
save_tensorboard_path = os.path.join(save_model_path, 'tensorboard_logs')
if not os.path.isdir(save_tensorboard_path):
    os.makedirs(save_tensorboard_path)      

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

other_ignored_in_training = True

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
#### ML parameters
batch = 32
epochs = 16 #100 #16
dense_neurons = 1024
dropout = 0.5
filters = 128 #y_train.shape[1] #

#------------------
# split between the training and the test set
train_test_split = 0.7
train_val_split = 0.80

#------------------
# data augmentation parameters
# n_steps = -2 # for pitch shift
# stretch_factor = 0.99 #If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
# scaling_factor = 0.1
# random_range = 0.1

#-----------------------------
# thresholding parameters
low_thresholds = [0.1,0.2,0.3]
high_thresholds = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

#------------------
# label munging parameters i.e. reading in audition or raven files
sep='\t'
engine = None
start_column = "t0File"
duration_column = "duration"
label_column = "entryName"
convert_to_seconds = True
label_for_other = "oth"
label_for_noise = "noise"
label_for_startstop = ['start', 'stop', 'skip', 'end']
multiclass_forbidden = True
ignore_others_as_category = True


#------------------
# call dictionary - 
# this is a dictionary containing as keys the category you want your ML algo to output
# and for each call category, how it is likely to be noted in the label column of the audition or raven file
# For example, Marker is usually for a close call.
# Note that these are regural expressions and are not case sensitive
call_types = {
    'cc' :["cc","Marker", "Marque"],
    'sn' :["sn","subm", "short","^s$", "s ", "s\*"],
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL", "so ", "so"],
    'al' :["al "," al ", " al","ALARM", "^al$"],
    'beep':["beep", "beeb", "beeo"],
    'synch':["sync", "sych"],
    # 'eating':['eating', 'eatimg'],
    'oth':["oth","other","lc", "lost","hyb","HYBRID","fu","sq", "seq","\+","ukn","unk","unknown",  "\#","\?"],
            #unsure calls
            # "x", "\%","\*", #noisy calls
            #"\$",
    'noise': ['start','stop','end','skip']
    }

# parameters that might be useful later that currently aren't dealt with
# we might want to make a separate category e.g. for focal and non-focal
# 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
# 'ukn':["ukn","unknown","UKN","UNKNOWN"]
# 'nf' :["nf","nonfoc","NONFOC"],
# 'noise':["x","X"]
# 'overlap':["%"]
# 'nf':["nf","nonfoc"]


# new  parameters for meerkat code
group_IDs = ["20190712_eating"] #["HM2017", "HM2019", "L2019"]
encoding = "ISO-8859-1" # used to be "utf-8"
columns_to_keep  = ['wavFileName', 'csvFileName', 'date', 'ind', 'group',
                    'callType', 'isCall', 'focalType', 'hybrid', 'noisy', 'unsureType']

# parameters for  noise augmentation and data generator
n_per_call = 3
mask_value = False
mask_vector = True


# parameters for the metrics
short_GT_removed = [0.0, 0.005, 0.015, 0.02, 0.025, 0.03, "none"] 
nonfoc_tags = ["NONFOC", "nf", "*"] 
start_labels = ['START','start']
stop_labels = ['END', 'STOP', 'stop']
skipon_labels = ['skipon', 'SKIPON']
skipoff_labels = ['skipoff', 'SKIPOFF']
headers = {'Label', 'Duration', 'Start', 'End'}