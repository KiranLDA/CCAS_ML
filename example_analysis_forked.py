#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 10:15:59 2021

@author: kiran
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#          IMPORT LIBRARIES
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

# import all the params for this model
import example_params

# add path to local functions
import os
os.chdir(github_dir)

# import own functions
import preprocess.preprocess_functions as pre
import postprocess.evaluation_metrics_functions_2 as metrics
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

# evaluate and plot 
import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import csv

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

# remove rows where the annotated noise is smaller than the window size otherwise the spectrogram we generate will inclue a call
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

# initialise the model class
model = rnn.BuildNetwork(x_train, num_calltypes, filters, gru_units, dense_neurons, dropout, mask_value)

# build the model
RNN_model = model.build_forked_masked_rnn()

# Adam optimiser
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile the model
RNN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

# Setup callbycks: learning rate / loss /tensorboard
early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)
loss = LossHistory()

# tensorboard
tensorboard = TensorBoard(log_dir = save_tensorboard_path,
                          histogram_freq=0,
                          write_graph=True,  # Show the network
                          write_grads=True   # Show gradients
                          )    


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

# #----------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------
# #               LOOP AND PREDICT OVER TEST FILES
# #----------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------


skipped_files = []

for file_ID in testing_filenames:
    # file_ID = testing_filenames[0]
    
    # subset the label table to only use that one file
    label_table = testing_label_table[testing_label_table['wavFileName'].isin([file_ID])].reset_index()
    # Find the file ID name i.e. remove the extention
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
                    
                    #generate the spectrogram
                    spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                            n_mels = n_mels, window='hann', 
                                                            fft_win= fft_win, fft_hop = fft_hop, 
                                                            normalise = True)
                    
                    # transpose it and put it in a format that works with the NN
                    spec = spectro.T
                    spec = spec[np.newaxis, ..., np.newaxis]  
                    
                    # generate a mask (as a placeholder) but don't mask anything as we are predicting and want to include other
                    mask = np.asarray([True for i in range(spectro.shape[1])])
                    mask = mask[np.newaxis,...]
                    
                    # generate the prediction
                    pred = RNN_model.predict([spec,mask])
                                       
                    # add this prediction to the stack that will be used to generate the predictions table
                    calltype_pred_list.append(np.squeeze(pred[0]))
                    callpresence_pred_list.append(np.squeeze(pred[1]))
                    
            # save the prediction stacks
            np.save( os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), calltype_pred_list)
            with open(os.path.join(save_pred_stack_test_path, file_ID + '_CALLTYPE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                for row in calltype_pred_list:
                    f.write(str(row) +"\n")
                    
            np.save( os.path.join(save_pred_stack_test_path, file_ID + '_CALLPRESENCE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), callpresence_pred_list)
            with open(os.path.join(save_pred_stack_test_path, file_ID + '_CALLPRESENCE_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                for row in callpresence_pred_list:
                    f.write(str(row) +"\n")
                
        # Loop through different sets of thresholds
        for low_thr in [0.1,0.2, 0.3]:
            for high_thr in [0.3,0.4,0.5,0.6,0.5,0.7,0.8,0.9,0.95]: 
                
                # make sure it doesnt generate a 0.00098982374957839486 type number
                low_thr = round(low_thr,2)                               
                high_thr = round(high_thr,2)
                
                # stop the loop if the low threshold is bigger than the high threshold
                if low_thr >= high_thr:
                    continue

                save_pred_table_filename = file_ID + "_CALLTYPE_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt"
                
                # if the file exists, pass to the next iteration of the loop
                if os.path.exists(os.path.join(save_pred_table_test_path, save_pred_table_filename)):
                    continue
                
                print("*****************************************************************") 
                print ("Low Threshold: " + str(low_thr))    
                print ("High Threshold: " + str(high_thr))  
                
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
                
                #in case the dataset was just noise, still create an empty placeholder to merge
                if len(detections) == 0:  
                    detections = pd.DataFrame(columns = ['category', 'start', 'end', 'scores'])
                
                # create an empty dataset
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

pred_tables = glob.glob(save_pred_table_test_path + "/*CALLTYPE_PRED_TABLE*.txt")
for file in pred_tables:
    df = pd.read_csv(file, delimiter=';') 
    # df = df.drop_duplicates(keep=False)
    df = df.loc[df['Label'] != 'Label']
    df.to_csv(file, header=True, index=None, sep=';', mode = 'w')


##############################################################################################
#
#            EVALUATE
#
##############################################################################################

#########################################################################
##  Create overall thresholds
#########################################################################

file_ID_list = [file_ID.split(".")[0] for file_ID in testing_filenames if file_ID not in skipped_files]
label_list =  [os.path.join(save_label_table_test_path,file_ID.split(".")[0]  + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]

# because of new file format, need to only keep certain columns
column_names = ["Label","Start","Duration","End"]
column_names.extend(list(testing_label_dict.keys()))
for file in label_list :
    df = pd.read_csv(file, delimiter=';') 
    # df = df.drop_duplicates(keep=False)
    df = df[column_names]
    df.to_csv(file, header=True, index=None, sep=';', mode = 'w')


for low_thr in [0.1,0.2,0.3]:
    for high_thr in [0.3,0.4,0.5,0.6,0.5,0.7,0.8,0.9,0.95]: 
        
        low_thr = round(low_thr,2)                               
        high_thr = round(high_thr,2) 
        
        if low_thr >= high_thr:
            continue
        
        pred_list = [os.path.join(save_pred_table_test_path,file_ID.split(".")[0]  + "_CALLTYPE_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt" ) for file_ID in file_ID_list ]
        evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
        # Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
        Prec, lenient_Prec, Rec, lenient_Rec, cat_frag, time_frag, cm, gt_indices, pred_indices, match, offset, call_match, pred_match, match2 = evaluation.main()
        
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
        
        # NEW METRICS
        lenient_Prec_filename  = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Lenient_Precision.csv'
        lenient_Rec_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Lenient_Recall.csv'
        call_match_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Call_Match.txt"
        pred_match_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Prediction_Match.txt"
        match2_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Matching_table.txt"
        
        lenient_Prec.to_csv(os.path.join(save_metrics_path, lenient_Prec_filename))        
        lenient_Rec.to_csv(os.path.join(save_metrics_path, lenient_Rec_filename))
        with open(os.path.join(save_metrics_path, call_match_filename), 'wb') as fp:
            pickle.dump(call_match, fp)  
        with open(os.path.join(save_metrics_path, pred_match_filename), 'wb') as fp:
            pickle.dump(pred_match, fp)  
        with open(os.path.join(save_metrics_path, match2_filename), 'wb') as fp:
            pickle.dump(match2, fp)  
        
        
        
        # save files
        Prec.to_csv( os.path.join(save_metrics_path, precision_filename))
        Rec.to_csv( os.path.join(save_metrics_path, recall_filename))
        cat_frag.to_csv( os.path.join(save_metrics_path, cat_frag_filename))
        time_frag.to_csv(os.path.join(save_metrics_path, time_frag_filename))
        cm.to_csv(os.path.join(save_metrics_path, confusion_filename))
        gt_indices.to_csv(os.path.join(save_metrics_path, gt_filename ))
        pred_indices.to_csv(os.path.join(save_metrics_path, pred_filename ))                  
        with open(os.path.join(save_metrics_path, match_filename), "wb") as fp:   #Picklin
            pickle.dump(match, fp)
        with open(os.path.join(save_metrics_path, timediff_filename), "wb") as fp:   #Pickling
            pickle.dump(offset, fp)    


#########################################################################
# plot overall confusion matrix
#########################################################################


# loop over the threaholds
for low_thr in [0.1,0.2,0.3]:
    for high_thr in [0.3,0.4,0.5,0.6,0.5,0.7,0.8,0.9,0.95]: 
        
        low_thr = round(low_thr,2)                               
        high_thr = round(high_thr,2) 
        
        if low_thr >= high_thr:
            continue

        confusion_filename = os.path.join(save_metrics_path, "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv')
        with open(confusion_filename, newline='') as csvfile:
            array = list(csv.reader(csvfile))
    
        df_cm = pd.DataFrame(array)#, range(6), range(6))    
        
        # get rid of the weird indentations and make rows and columns as names
        new_col = df_cm.iloc[0] # grab the first row for the header
        df_cm = df_cm[1:] # take the data less the header row
        df_cm.columns = new_col # set the header row as the df header    
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
        
        df_cm = df_cm.apply(pd.to_numeric)
                
        # Raw confusion matrix
        df_cm = df_cm[list(testing_label_dict.keys())]
        df_cm = df_cm.reindex(list(testing_label_dict.keys()))
        
        # plot raw
        ax = plt.axes()
        sn.set(font_scale=1.1) # for label size
        sn.heatmap((df_cm+1), annot=df_cm, fmt='g',norm = LogNorm(), annot_kws={"size": 10}, ax= ax) # font size
        ax.set_title(str(low_thr) + "-" + str(high_thr) )
        plt.savefig(os.path.join(save_metrics_path, "Confusion_mat_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
        plt.show()

        # Recall confusion matrix
        df_recall = df_cm.div(df_cm.sum(axis=1), axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
        
        # plot recall
        ax = plt.axes()
        sn.set(font_scale=1.1) # for label size
        sn.heatmap((df_recall), annot=True, fmt='g', annot_kws={"size": 10}, ax= ax) # font size
        ax.set_title(str(low_thr) + "-" + str(high_thr) )
        plt.savefig(os.path.join(save_metrics_path, "Confusion_mat_recall_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
        plt.show()        
        
        # Proportion of calls for confusion matrix
        call_len = list()
        for i in testing_label_dict.keys():
            call_len.append(testing_label_dict[i].shape[0])
        # add noise at the end
        call_len[-1] = df_cm.sum(axis=1)[-1]
        
        # plot proportion of calls
        df_prop = df_cm.div(call_len, axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
        ax = plt.axes()
        sn.set(font_scale=1.1) # for label size
        sn.heatmap((df_prop), annot=True, fmt='g', annot_kws={"size": 10}, ax= ax) # font size
        ax.set_title(str(low_thr) + "-" + str(high_thr) )
        plt.savefig(os.path.join(save_metrics_path, "Confusion_mat_prop_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
        plt.show()

