#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:31:12 2021

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
import model.specgen_batch_generator as bg
import model.network_class as rnn
# import postprocess.visualise_prediction_functions as pp
from model.callback_functions import LossHistory



# import normal packages used in pre-processing
import numpy as np
import librosa
import warnings
import ntpath
import os
from itertools import compress  
from random import random, shuffle
from math import floor
import statistics
import glob

# plotting
import matplotlib.pyplot as plt

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



#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS - will likely put them in another directory
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


other_ignored_in_training = True
run_name = "NoiseAugmented_NotWeighted_MaskedOther_Forked"

#------------------
# File paths
#------------------
# label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
#             "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
label_dirs =["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/working/processed/acoustic/total_synched_call_tables"]


# audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
#              "/media/kiran/Kiran Meerkat/Meerkat data 2019"]
audio_dirs =["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/archive/rawdata/MEERKAT_RAW_DATA"]

acoustic_data_path = ["/home/kiran/Documents/MPI-Server/EAS_shared/meerkat/working/processed/acoustic"]


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

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          PARAMETERS
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

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
train_val_split = 0.80

#------------------
# data augmentation parameters
n_steps = -2 # for pitch shift
stretch_factor = 0.99 #If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
scaling_factor = 0.1
random_range = 0.1


min_scaling_factor= 0.5
max_scaling_factor= 1.1
#-----------------------------
# thresholding parameters
low_thr = 0.2

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


# new  parameters for meerkat code
group_IDs = ["HM2017", "HM2019", "L2019"]
encoding = "ISO-8859-1" # used to be "utf-8"
columns_to_keep  = ['wavFileName', 'csvFileName', 'date', 'ind', 'group',
                    'callType', 'isCall', 'focalType', 'hybrid', 'noisy', 'unsureType']

# parameters for 

min_scaling_factor = 0.1
max_scaling_factor = 0.5
n_per_call = 3
mask_value = False#1000
mask_vector = True
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#             1 - PREPROCESSING - SETTING UP DIRECTORIES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# Compile all the synched label files together
labels_all = pd.DataFrame()
for directory in label_dirs:
    for group in group_IDs:
        temp = pd.read_csv(os.path.join(directory, group +"_ALL_CALLS_SYNCHED.csv"), sep=sep,
                       header=0, engine = engine, encoding = encoding) 
        temp["group"] = group
        labels_all = pd.concat([labels_all, temp]) 
        del temp
labels_all = labels_all.reset_index(drop = True)
labels_all = labels_all[~labels_all.wavFileName.str.contains('SOUNDFOC')]

#Summary data
# labels_all.groupby(['ind','group']).size().reset_index().rename(columns={0:'count'})
# labels_all.groupby(['ind','group', 'date']).size().reset_index().rename(columns={0:'count'})
# labels_all.groupby(['ind', 'date']).size().reset_index().rename(columns={0:'count'})


# subset all the audio files that we should use in the analysis (i.e. not focal follow data)
audio_files = list(set(labels_all["wavFileName"]))
audio_filenames = list(compress(audio_files, ["SOUNDFOC" not in filei for filei in audio_files]))

# subset all the audio files that we should use in the analysis (i.e. not focal follow data)
label_files = list(set(labels_all["csvFileName"]))
label_filenames = list(compress(label_files, ["SOUNDFOC" not in filei for filei in label_files]))

# get the file IDS without all the extentions (used later for naming)
all_filenames = [audio_filenames[i].split(".")[0] for i in range(0,len(audio_filenames))]


#----------------------------------------------------------------------------------
# find all the paths to the files
#----------------------------------------------------------------------------------

# find all the labels
EXT = "*.csv"
label_filepaths = []
for PATH in acoustic_data_path :
      label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])
EXT = "*.CSV"
for PATH in acoustic_data_path :
      label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])

# find all audio paths (will be longer than label path as not everything is labelled)
audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
      audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       2 - Create a massive label table
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# Create the label table
label_table = pre.create_meerkat_table(labels_all, call_types, sep,
                                       start_column, duration_column, columns_to_keep,
                                       label_column, convert_to_seconds, 
                                       label_for_other, label_for_noise, engine,
                                       multiclass_forbidden)

# estimate the average beep length because many of them are not annotated in the data
avg_beep = round(statistics.mean(label_table.loc[label_table["beep"],"Duration"].loc[label_table.loc[label_table["beep"],"Duration"]>0]),3)
label_table.loc[(label_table["beep"].bool and label_table["Duration"] == 0.) ==True, "Duration"] = avg_beep
label_table.loc[(label_table["beep"].bool and label_table["Duration"] == avg_beep) ==True, "End"] += avg_beep

# add wav and audio paths
label_table["wav_path"] = label_table['wavFileName'].apply(lambda x: [pathi for pathi in audio_filepaths if x in pathi][0])
label_table["label_path"] = label_table['csvFileName'].apply(lambda x: [pathi for pathi in label_filepaths if x in pathi][0])

# make sure these paths are added to the noise table too
columns_to_keep.append("wav_path")
columns_to_keep.append("label_path")

# create the matching noise table
noise_table = pre.create_noise_table(label_table, call_types, label_for_noise, label_for_startstop, columns_to_keep)#, '\$'])
# remove rows where the annotated noise is smaller than the window size
noise_table = noise_table.drop(noise_table[noise_table["Duration"] < spec_window_size].index)


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
    with open(os.path.join(save_model_path, "validation_files_used.txt")) as f:
        content = f.readlines()    
    validation_filenames = [x.strip() for x in content] # remove whitespace characters like `\n` at the end of each line

# otherwiss create the training and testing files
else: 
    # randomise the order of the files
    file_list = audio_filenames #all_filenames
    shuffle(file_list)
    
    # randomly divide the files into those in the training and test datasets
    split_index = floor(len(file_list) * train_test_split)
    training_files = file_list[:split_index]
    testing_filenames = file_list[split_index:]
    
    split_index = floor(len(training_files) * train_val_split )
    training_filenames = training_files[:split_index]
    validation_filenames = training_files[split_index:]

    # save a copy of the training and testing diles
    with open(os.path.join(save_model_path, "training_files_used.txt"), "w") as f:
        for s in training_filenames:
            f.write(str(s) +"\n")
    with open(os.path.join(save_model_path, "testing_files_used.txt"), "w") as f:
        for s in testing_filenames:
            f.write(str(s) +"\n")
    with open(os.path.join(save_model_path, "validation_files_used.txt"), "w") as f:
        for s in validation_filenames:
            f.write(str(s) +"\n")

#-------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# 2.1. Generate and save the training files
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# separate out the training and test sets for analysis
training_label_table = label_table[label_table['wavFileName'].isin(training_filenames)]
testing_label_table = label_table[label_table['wavFileName'].isin(testing_filenames)]
validation_label_table = label_table[label_table['wavFileName'].isin(validation_filenames)]

training_noise_table = noise_table[noise_table['wavFileName'].isin(training_filenames)]
testing_noise_table = noise_table[noise_table['wavFileName'].isin(testing_filenames)]
validation_noise_table = noise_table[noise_table['wavFileName'].isin(validation_filenames)]

# Compile data into a format that the data generator can use
training_label_dict = dict()
for label in call_types: 
    training_label_dict[label] = training_label_table.loc[training_label_table[label] == True, ["Label", "Start", "Duration","End","wav_path","label_path"]]
training_label_dict[label_for_noise] = training_noise_table[["Label", "Start", "Duration","End","wav_path","label_path"]]

# Compile data into a format that the data generator can use
testing_label_dict = dict()
for label in call_types: 
    testing_label_dict[label] = testing_label_table.loc[testing_label_table[label] == True, ["Label", "Start", "Duration","End","wav_path","label_path"]]
testing_label_dict[label_for_noise] = testing_noise_table[["Label", "Start", "Duration","End","wav_path","label_path"]]

# Compile data into a format that the data generator can use
validation_label_dict = dict()
for label in call_types: 
    validation_label_dict[label] = validation_label_table.loc[validation_label_table[label] == True, ["Label", "Start", "Duration","End","wav_path","label_path"]]
validation_label_dict[label_for_noise] = validation_label_table[["Label", "Start", "Duration","End","wav_path","label_path"]]


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       2.5. BATCH GENERATOR FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

# initiate the data generator
train_generator = bg.ForkedDataGenerator(training_label_dict,
                                         training_label_table, 
                                         spec_window_size,
                                         n_mels, 
                                         window, 
                                         fft_win , 
                                         fft_hop , 
                                         normalise,
                                         label_for_noise,
                                         label_for_other,
                                         min_scaling_factor,
                                         max_scaling_factor,
                                         n_per_call,
                                         other_ignored_in_training,
                                         mask_value,
                                         mask_vector)

    
# generate an example spectrogram and label  
spec, label, callmat, mask = train_generator.generate_example("sn", 0, True)

#----------------------------------------------------------------------------------------------------
# Do some plotting to ensure it makes sense

# set the labels
label_list = list(call_types.keys())
if other_ignored_in_training:
    label_list.remove(label_for_other)
  
# plot spectrogram
plt.figure(figsize=(10, 10))
plt.subplot(411)
yaxis = range(0, np.flipud(spec).shape[0]+1)
xaxis = range(0, np.flipud(spec).shape[1]+1)
librosa.display.specshow(spec,  y_axis='mel', x_coords = label.columns)#, x_axis= "time",sr=sr, x_coords = label.columns)
plt.ylabel('Frequency (Hz)')
plt.clim(-35, 35)
plt.colorbar(format='%+2.0f dB')

# plot LABEL
plt.subplot(412)
xaxis = range(0, np.flipud(label).shape[1]+1)
yaxis = range(0, np.flipud(label).shape[0]+1)
plt.yticks(np.arange(0.5, len(label_list)+0.5 ,1 ),reversed(label_list))
plt.xticks(np.arange(0, np.flipud(label).shape[1]+1,50),
           list(label.columns[np.arange(0, np.flipud(label).shape[1]+1,50)]))
plt.pcolormesh(xaxis, yaxis, np.flipud(label))
plt.xlabel('Time (s)')
plt.ylabel('Calltype')
plt.colorbar(label="Label")


# plot call matrix
plt.subplot(413)
xaxis = range(0, np.flipud(label).shape[1]+1)
yaxis = range(0, np.flipud(callmat).shape[0]+1)
plt.yticks(np.arange(0.5, callmat.shape[0]+0.5 ,1 ), reversed(callmat.index.values))
plt.xticks(np.arange(0, np.flipud(label).shape[1]+1,50),
           list(label.columns[np.arange(0, np.flipud(label).shape[1]+1,50)]))
plt.pcolormesh(xaxis, yaxis, np.flipud(callmat))
plt.xlabel('Time (s)')
plt.ylabel('Call / No Call')
plt.colorbar(label="Label")
# 

plt.subplot(414)
# plot the mask
if mask_vector == True:
    test = np.asarray([int(x == True) for x in mask])
    # test = np.hstack(test)
    test = test[np.newaxis, ...]
    # plt.imshow(mask, aspect='auto', cmap=plt.cm.gray)
    xaxis = range(0, np.flipud(label).shape[1]+1)
    yaxis = range(0, np.flipud(test).shape[0]+1)
    # yaxis = range(0, 1)
    # plt.yticks(np.arange(0.5, 1 ,1 ), "oth")
    plt.yticks(np.arange(0.5, test.shape[0]+0.5 ,1 ), reversed(["oth"]))
    plt.xticks(np.arange(0, np.flipud(label).shape[1]+1,50),
               list(label.columns[np.arange(0, np.flipud(label).shape[1]+1,50)]))
    plt.pcolormesh(xaxis, yaxis, np.flipud(test))
    plt.xlabel('Time (s)')
    plt.ylabel('True/False')
    plt.colorbar(label="Label")
    plt.clim(0, 1)
    plt.show()
else:

    xaxis = range(0, np.flipud(mask).shape[1]+1)
    yaxis = range(0, np.flipud(mask).shape[0]+1)
    # plt.yticks(np.arange(0.5, np.flipud(mask).shape[0]+0.5 ,1 ),reversed(label_list))
    plt.xticks(np.arange(0, np.flipud(mask).shape[1]+1,50),
               list(label.columns[np.arange(0, np.flipud(mask).shape[1]+1,50)]))
    plt.pcolormesh(xaxis, yaxis, np.flipud(mask))
    plt.xlabel('Time (s)')
    plt.ylabel('Spectrogram Channels')
    plt.colorbar(label="Other or not")
    plt.show()



#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       3. CONSTRUCT THE RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# get a batch to estimate rnn parameters
x_train, y_train = train_generator.__next__()#__getitem__(0)
# x_train, y_train = val_generator.__next__()#__getitem__(0)
# print(x_train[0].shape)
# print(x_train[1].shape)
# print(y_train[0].shape)
# print(y_train[1].shape)



# initial parameters
num_calltypes = y_train[0].shape[2]
gru_units = y_train[0].shape[1] 

# initialise the training data generator and validation data generator
train_generator = bg.ForkedDataGenerator(training_label_dict,
                                         training_label_table, 
                                         spec_window_size,
                                         n_mels, 
                                         window, 
                                         fft_win , 
                                         fft_hop , 
                                         normalise,
                                         label_for_noise,
                                         label_for_other,
                                         min_scaling_factor,
                                         max_scaling_factor,
                                         n_per_call,
                                         other_ignored_in_training,
                                         mask_value,
                                         mask_vector)

val_generator = bg.ForkedDataGenerator(validation_label_dict,
                                       validation_label_table, 
                                       spec_window_size,
                                       n_mels, 
                                       window, 
                                       fft_win , 
                                       fft_hop , 
                                       normalise,
                                       label_for_noise,
                                       label_for_other,
                                       min_scaling_factor,
                                       max_scaling_factor,
                                       n_per_call,
                                       other_ignored_in_training,
                                       mask_value,
                                       mask_vector)





#--------------------------------------------
# Construct the RNN
#--------------------------------------------
# import model.network_class as rnn
# import importlib
# importlib.reload(rnn)
# mask_vector = False

model = rnn.BuildNetwork(x_train, num_calltypes, filters, gru_units, dense_neurons, dropout, mask_value)

RNN_model = model.build_forked_masked_rnn()

# Adam optimiser
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
RNN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Setup callbycks: learning rate / loss /tensorboard
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
loss = LossHistory()

#tensorboardTrue
tensorboard = TensorBoard(# Write to logs directory, e.g. logs/30Oct-05:00
                          log_dir = save_tensorboard_path, #"/media/kiran/D0-P1/animal_data/meerkat/NoiseAugmented_NoOther/trained_model/tensorboard_logs", #"logs/{}".format(time.strftime('%d%b-%H%M')),        
                          histogram_freq=0,
                          write_graph=True,  # Show the network
                          write_grads=True   # Show gradients
                          )    

# epochs = 3
# fit model
RNN_model.fit_generator(train_generator, 
                        steps_per_epoch = train_generator.__len__(),
                        epochs = epochs,
                        callbacks = [early_stopping, reduce_lr_plat, loss, tensorboard],
                        validation_data = val_generator,
                        validation_steps = val_generator.__len__())

 

# save the model
date_time = datetime.datetime.now()
date_now = str(date_time.date())
time_now = str(date_time.time())

sf = os.path.join(save_model_path, run_name+ "_" + date_now + "_" + time_now)
if not os.path.isdir(sf):
        os.makedirs(sf)

RNN_model.save(sf + '/savedmodel' + '.h5')

# sf = "/media/kiran/D0-P1/animal_data/meerkat/NoiseAugmented_NotWeighted_MaskedOther_Forked/trained_model/NoiseAugmented_NotWeighted_MaskedOther_Forked_2021-04-09_19:06:51.536406"
# RNN_model =load_model("/media/kiran/D0-P1/animal_data/meerkat/NoiseAugmented_NotWeighted_MaskedOther_Forked/trained_model/NoiseAugmented_NotWeighted_MaskedOther_Forked_2021-04-09_19:06:51.536406/test_savedmodel.h5")



# #----------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------
# #               LOOP AND PREDICT OVER TEST FILES
# #----------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------


skipped_files = []

for file_ID in testing_filenames[1:len(testing_filenames)]:
    # file_ID = testing_filenames[1]

    label_table = testing_label_table[testing_label_table['wavFileName'].isin([file_ID])].reset_index()
    file_ID = file_ID.split(".")[0] 
    # find the matching audio for the label data
    audio_path = label_table["wav_path"][0]   
    
    print("*****************************************************************")   
    print("*****************************************************************") 
    print ("File being processed : " + audio_path)    
    
    # find the start and stop  of the labelling periods (also using skipon/skipoff)
    loop_table = label_table.loc[label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False), ["Label","Start"]]
    loop_times = list(loop_table["Start"])
    
    # Make sure that the file contains the right number of start and stops, otherwise go to the next file
    if len(loop_times)%2 != 0:
        print("!!!!!!!!!!!!!!!!")
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + audio_path)
        skipped_files.append(file_ID)
        continue 
    
    if len(loop_times) == 0:
        print("!!!!!!!!!!!!!!!!")
        warnings.warn("There is a missing start or stop in this file and it has been skipped: " + audio_path)
        skipped_files.append(file_ID)
        continue 
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
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
        if os.path.exists(os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy')): 
            calltype_pred_list = np.load( os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy')) 
            callpresence_pred_list = np.load( os.path.join(save_pred_stack_test_path, file_ID + '_CALLPRESENCE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'))
       
        # if not, generate it
        else:
        
            calltype_pred_list = []
            callpresence_pred_list = []
    
            for spectro_slide in np.arange(fromi, toi, slide):
                
                # spectro_slide = fromi
                start = round(spectro_slide,3)
                stop = round(spectro_slide + spec_window_size, 3)
                

                # ignore cases where the window is larger than what is labelled (e.g. at the end)
                if stop <= toi:
                    
                    spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                            n_mels = n_mels, window='hann', 
                                                            fft_win= fft_win, fft_hop = fft_hop, 
                                                            normalise = True)
                    
                    # label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                    #                                         stop, label_for_noise)
                    
                    # Load the spectrogram
                    spec = spectro.T
                    spec = spec[np.newaxis, ..., np.newaxis]  
                    
                    mask = np.asarray([True for i in range(spectro.shape[1])])
                    mask = mask[np.newaxis,...]
                    
                    # generate the prediction
                    pred = RNN_model.predict([spec,mask])
                                       
                    # add this prediction to the stack that will be used to generate the predictions table
                    calltype_pred_list.append(np.squeeze(pred[0]))
                    callpresence_pred_list.append(np.squeeze(pred[1]))
                    
            # save the prediction list  
            np.save( os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), calltype_pred_list)
            with open(os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                for row in calltype_pred_list:
                    f.write(str(row) +"\n")
                    
            np.save( os.path.join(save_pred_stack_test_path, file_ID + '_CALLPRESENCE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), callpresence_pred_list)
            with open(os.path.join(save_pred_stack_test_path, file_ID + '_CALLPRESENCE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                for row in callpresence_pred_list:
                    f.write(str(row) +"\n")
                
        for low_thr in [0.2]:#[0.1,0.3]:
            for high_thr in [0.7,0.8,0.9,0.97]: #[0.5,0.7,0.8,0.9,0.95]: 

                
                low_thr = round(low_thr,2)                               
                high_thr = round(high_thr,2)

                save_pred_table_filename = file_ID + "_CALLTYPE_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt"
                
                #----------------------------------------------------------------------------
                # Compile the predictions for each on/off labelling chunk
                detections = ppm.merge_p(probabilities = calltype_pred_list, 
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
                
                
                
    




               
                
        
# '''
# # load the saved file
# with open(os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK.txt')) as f:
#     content = f.readlines()
# # remove whitespace characters like `\n` at the end of each line
# pred_list = [x.strip() for x in content] 


# #or
# pred_list = np.load( os.path.join(save_pred_stack_test_path, file_ID + '_PRED_STACK.npy'))
   
# '''
        
# save the files that were skipped
print(skipped_files)

# save a copy of the training and testing diles
with open(os.path.join(save_model_path, "skipped_testing_files.txt"), "w") as f:
    for s in skipped_files:
        f.write(str(s) +"\n")
       
##############################################################################################
# Loop through tables and remove duplicates of rows (bevause files are created through appending)

pred_tables = glob.glob(save_pred_table_test_path+ "/*CALLTYPE_PRED_TABLE*.txt")
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
    for high_thr in [0.7,0.8,0.9,0.97]: #[0.5,0.7,0.8,0.9,0.95]: 
        
        low_thr = round(low_thr,2)                               
        high_thr = round(high_thr,2) 
        
        pred_list = [os.path.join(save_pred_table_test_path,file_ID + "_CALLTYPE_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt" ) for file_ID in file_ID_list ]
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
for low_thr in [0.2]:#[0.1,0.3]:
    for high_thr in [0.5,0.7,0.9]: #[0.5,0.7,0.8,0.9,0.95]: 
        
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


