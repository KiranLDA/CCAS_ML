#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:27:13 2021

@author: kiran
"""

import os




#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

other_ignored_in_training = True

#------------------
#### ML parameters
batch = 32
epochs = 200 #16 #16
dense_neurons = 1024
dropout = 0.5
filters = 128 #y_train.shape[1] #

#------------------
# split between the training and the test set
train_test_split = 0.80
train_val_split = 0.80

#------------------
# parameters for noise augmentation and data generator
n_per_call = 3
mask_value = False
mask_vector = True
min_scaling_factor = 0.3
max_scaling_factor = 0.8
# n_steps = -2 # for pitch shift
# stretch_factor = 0.99 #If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
# random_range = 0.1

#------------------
# new  parameters for meerkat code (since on the server)
group_IDs = ["HM2017", "HM2019", "L2019"]
encoding = "ISO-8859-1" # used to be "utf-8"
columns_to_keep  = ['wavFileName', 'csvFileName', 'date', 'ind', 'group',
                    'callType', 'isCall', 'focalType', 'hybrid', 'noisy', 'unsureType']

#------------------
# rolling window parameters for spectrogram generation
spec_window_size = 1
slide = 0.5

#------------------
# fast fourier parameters for mel spectrogram generation
fft_win = 0.01 #0.03
fft_hop = fft_win/2
n_mels = 30 #128
window = "hann"
normalise = True

#-----------------------------
# thresholding parameters
# low_thr = 0.2

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
    'beep':["beep", "beeb"],
    'synch':["sync"],
    'oth':["oth","other","lc", "lost","hyb","HYBRID","fu","sq", "seq","\+","ukn","unk","unknown",  "\#","\?"],
            #unsure calls
            # "x", "\%","\*", #noisy calls
            #"\$",
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
# evaluation parameters
no_call = set(["noise", "beep", "synch"])
true_call= set(set(call_types.keys()).difference(no_call))

# will go in params
eval_analysis = "call_type_by_call_type" #"normal"
#true_call= set(list(call_types.keys()).difference(no_call))

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#                 File paths
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
#             "/home/kiran/Documents/ML/meerkats_2019/labels_csv"] #2019 meerkats
label_dirs =["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/working/processed/acoustic/total_synched_call_tables"]


# audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
#              "/media/kiran/Kiran Meerkat/Meerkat data 2019"]
audio_dirs =["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/archive/rawdata/MEERKAT_RAW_DATA"]

acoustic_data_path = ["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/working/processed/acoustic"]

# 
results_dir = '/media/kiran/D0-P1/animal_data/meerkat'
run_name = "EXAMPLE_NoiseAugmented_"+ str(min_scaling_factor)+"_" +str(max_scaling_factor)+"_NotWeighted_MaskedOther_Forked"


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
test_path = os.path.join(save_data_path, 'test_data')
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
        
save_metrics_path_eval = os.path.join(save_metrics_path, eval_analysis)
if not os.path.isdir(save_metrics_path_eval):
    os.makedirs(save_metrics_path_eval)


# Model folder
save_model_path = os.path.join(save_data_path, 'trained_model')
if not os.path.isdir(save_model_path):
    os.makedirs(save_model_path)
        
   
