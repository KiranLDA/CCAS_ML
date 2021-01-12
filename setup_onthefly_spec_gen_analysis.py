#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:15:51 2020

@author: kiran
"""
import os
os.chdir("/home/kiran/Documents/github/CCAS_ML")
# sys.path.append("/home/kiran/Documents/github/CCAS_ML")

import preprocess.preprocess_functions as pre
import pandas as pd
# for every training file get the label table


from params import *
# from preprocess.initialise_params import InitialiseParams
# initialising = InitialiseParams("/home/kiran/Documents/github/CCAS_ML/params.txt")
# initialising.call_types
# initialising.items()
import postprocess.cleaning as cl
# 


#----------------------------------------------------------------------------------

import os

other_ignored_in_training = True
run_name = "NoiseAugmented_ProportionallyWeighted_NoOther"

#------------------
# File paths
#------------------
label_dirs = ["/home/kiran/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2019/labels_CSV/labels_csv_20210107"]
audio_dirs= ["/home/kiran/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2017",
             "/home/kiran/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2019"]

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
#### ML parameters
batch = 32
epochs = 100 #16 #16
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
multiclass_forbidden = True

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
    'al' :["al "," al ", " al","ALARM", "^al$"],
    'beep':["beep", "beeb"],
    'synch':["sync"],
    'oth':["oth","other","lc", "lost",
           "hyb","HYBRID","fu","sq", "seq","\+",
           "ukn","unk","unknown",          
           # "x", "\%","\*", #noisy calls
           #"\$",
            "\#","\?" #unsure calls
           ],
    'noise':['start','stop','end','skip', '\$']
    }

# parameters that might be useful later that currently aren't dealt with
# we might want to make a separate category e.g. for focal and non-focal
# 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
# 'ukn':["ukn","unknown","UKN","UNKNOWN"]
# 'nf' :["nf","nonfoc","NONFOC"],
# 'noise':["x","X"]
# 'overlap':["%"]
# 'nf':["nf","nonfoc"]
# '\$' #is a call incorrectly found by ari's code

# get rid of the focal follows (going around behind the meerkat with a microphone)
to_rm = ["SOUNDFOC", "PROCESSED", "LABEL" , "label", "_SS"]




#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#             1 - PREPROCESSING 
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


# Find the input data ie. the specific filenames
#-----------------------------------------------------------------
label_filenames, label_filepaths, audio_filenames, audio_filepaths = cl.find_audio_and_label_files(label_dirs, audio_dirs, to_rm)


# # Split the label filenames into training and test files
# #----------------------------------------------------------------------------------------------------

# If the training and testing files exists then load them, otherwise create them
training_filenames, testing_filenames = cl.split_into_train_testsets(save_model_path, label_filenames, train_test_split)


mega_table = pd.DataFrame()
mega_noise_table = pd.DataFrame()

# training_filenames = label_filenames
# Start the loop by going over every single labelled file id
for file_ID in training_filenames:
    # file_ID = label_filenames[2]
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
    
    # only generate training files if they don't already exist    
    # if not os.path.exists(os.path.join(save_label_table_train_path, save_label_table_filename)):
        
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    
    # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    
    print ("File being processed : " + label_path)    
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, 
                                   label_for_other, label_for_noise, engine, multiclass_forbidden)
    
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
    label_table.loc[label_table["beep"] == True, "End"] += 0.04
    
    # don't save while testing    
    # label_table.to_csv(os.path.join(save_label_table_train_path, save_label_table_filename), header=True, index=None, sep=';')
    
    # #save the label tables with other, but for the purpose of labelling, remove other
    # if other_ignored_in_training:
    #     label_table = label_table[label_table[label_for_other] == False]
    #     label_table= label_table.reset_index(drop=True)
    
    noise_table = pre.create_noise_table(label_table, label_for_noise, label_for_startstop=['start', 'stop', 'skip', 'end', '\$'])
    
    label_table["File"] = file_ID
    noise_table["File"] = file_ID
    mega_table = pd.concat([mega_table, label_table])
    mega_noise_table = pd.concat([mega_noise_table, noise_table])

call_table_dict = {}
# create individual tables for all calls
for label in call_types: 
    call_table_dict[label] = mega_table.loc[mega_table[label] == True, ["Label", "Start", "Duration","End","File"]]

    
call_table_dict[label_for_noise]=mega_noise_table[["Label", "Start", "Duration","End","File"]]



call_table_dict

#------------------------------------------------------------------------------
# data augmentation parameters
#------------------------------------------------------------------------------

sample_size = pd.DataFrame()

for label in call_table_dict: 
    sample_size = sample_size.append(pd.DataFrame([[label,len(call_table_dict[label]),
                                                    sum(call_table_dict[label]["Duration"])]],
                                                  columns= ["label", "sample_size", "duration"]))

sample_size["multiply_by"] = round(sample_size.loc[sample_size ["label"]=="cc", "sample_size"] / sample_size["sample_size"])
sample_size["data_augment"] = sample_size["multiply_by"]  - 1


print(sample_size)
sample_size["data_augment"] = sample_size["multiply_by"]  - 1
sample_size.loc[sample_size ["label"]=="cc", "data_augment"] = 0.5
sample_size["noise_time_needed"] = sample_size["sample_size"]*sample_size["data_augment"]*spec_window_size






##  NEW NOISE AUGMENTATION - WILL PUT INTO FUNCTION LATER
call_table_dict
wav_filepaths
mega_table
wav_filepaths = audio_filepaths
random_range # = call_offset
# # randomly choose a spectrogram 
# call_spec = random.choice([x for x in spec_filepaths if calltype in x])#glob.glob(folder + "/*" + calltype +".npy")
# # keep only the file name
# call_spec = os.path.basename(call_spec) 
# # keep only the general name of the file so it can be linked to the corresponding .wav
# call_bits = re.split("_SPEC_", call_spec)
# file_ID = call_bits[0] 
calltype = "sn"
index = 1




file_ID = call_table_dict[calltype].iloc[index]["File"]
# find the wav
call_wav_path = [s for s in wav_filepaths if file_ID in s][0]
#load the wave
y, sr = librosa.load(call_wav_path, sr=None, mono=False)


# find the corresponding labels
label_table = mega_table.loc[mega_table["File"]==file_ID]
#save the label tables with other, but for the purpose of labelling, remove other
if other_ignored_in_training:
    label_table = label_table[label_table[label_for_other] == False]
    label_table= label_table.reset_index(drop=True)

if sample_size.loc[sample_size["label"] == calltype, "data_augment"] >= 1:
    for i in range(sample_size.loc[sample_size["label"] == calltype, "data_augment"]):

        #randomise the start a little so the new spectrogram will be a little different from the old
        call_start = round(float(call_table_dict[calltype].iloc[index]["Start"]+np.random.uniform(-random_range, random_range, 1)), 3)
        call_stop = round(call_start + spec_window_size,3 )
        
        start_lab = int(round(sr * decimal.Decimal(call_start),3))
        stop_lab =  int(round(sr * decimal.Decimal(call_stop),3))
        #suset the wav
        data_subset = np.asfortranarray(y[start_lab:stop_lab])
        

        # randomly choose a noise file - same as above, only with noise
        noise_event =  mega_noise_table.sample()#random.choice([x for x in noise_filepaths if label_for_noise in x])#glob.glob(folder + "/*" + calltype +".npy")
        noise_ID = noise_event.iloc[0]["File"]
        noise_wav_path = [s for s in wav_filepaths if noise_ID in s][0]
        noise_start = round(float(np.random.uniform(noise_event.iloc[0]["Start"], (noise_event.iloc[0]["End"]-spec_window_size), 1)), 3)
        noise_stop = round(noise_start + spec_window_size,3 )
        y_noise, sr = librosa.load(noise_wav_path, sr=None, mono=False)
        start = int(round(sr * decimal.Decimal(noise_start),3))
        stop =  int(round(sr * decimal.Decimal(noise_stop),3))
        noise_subset = np.asfortranarray(y_noise[start:stop])


##### got this far

# combine the two
augmented_data = data_subset + noise_subset * scaling_factor
# generate spectrogram
augmented_spectrogram = generate_mel_spectrogram(augmented_data, sr, 0, spec_window_size, 
                                                  n_mels, window, fft_win , fft_hop , normalise)
# generate label
augmented_label = create_label_matrix(label_table, augmented_spectrogram,
                                      call_types, call_start, call_stop, 
                                      label_for_noise)

# find out what the label is for this given window so that later we can choose the label/test set in a balanced way
file_label = list(augmented_label.index.values[augmented_label.where(augmented_label > 0).sum(1) > 1])
if len(file_label) > 1 and label_for_noise in file_label:
    file_label.remove(label_for_noise)
category = '_'.join(file_label)
        
# Save these files
aug_spec_filename = file_ID + "_SPEC_" + str(call_start) + "s-" + str(call_stop) + "s_NOISE_AUGMENTED_" + category + ".npy"
aug_mat_filename = file_ID + "_MAT_" + str(call_start) + "s-" + str(call_stop) + "s_NOISE_AUGMENTED_" + category + ".npy"
    

return augmented_data, augmented_spectrogram, augmented_label, aug_spec_filename, aug_mat_filename










'''
model.fit_generator(data_generator_for_classif(clips_dir = clips_dir,batch_size = batch_size), 
                    epochs=epochs, use_multiprocessing=True, 
                    workers=16, 
                    steps_per_epoch=steps_per_epoch)
'''