#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 11:35:57 2020
These are functions for looking at prediction
@author: kiran
"""

import random
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

def predict_label_from_random_spectrogram(RNN_model, calltype, save_spec_path, save_mat_path):
    '''
    This code finds a random spectrogram of a give calltype, predicts on th sprectrogram 
    and returns the label to compare the prediction to.
    
    IN:
        RRN_model: 
            trained rnn object already loaded 
            i.e. keras.models.load_model("filepath/filename.h5")
        calltype:
            string specifying which calltype to look for in the filename. 
            e.g. "cc" for a close call in meerkats
        save_spec_path:
            location of saved spectrogram
        save_mat_path:
            location of saved label
    OUT:
        spec:
            the loaded spectrogram
        label:
            the model label
        pred:
            the prediction from the model
        
    '''
    # subset spectrograms by calltype
    filei = [i for i in os.listdir(save_spec_path)  if calltype in i] 
    idxi = [filei.index(i) for i in filei]
    
    # select a random file
    i= random.choice(idxi) 
    spec_filename = filei[i]
    
    # Load the spectrogram
    spec = np.asarray(np.load(os.path.join(save_spec_path, spec_filename)).T)
    spec = spec[np.newaxis, ..., np.newaxis]  
    
    # predict using trained model
    pred = RNN_model.predict(spec)
    
    #Find matching label file and load it
    label_filename = '_MAT_'.join(  spec_filename.split("_SPEC_"))
    label = np.load(os.path.join(save_mat_path, label_filename))
    
    return spec, label, pred


#********************************************************************************



def plot_spec_label_pred(spec, label, pred, label_list):
    
    '''
    this code takes the output from predict_label_from_random_spectrogram()
    and plots it
    
    IN:
        spec:
            the loaded spectrogram
        pred:
            the prediction from the model
        label:
            the model label
        label_list:
            a list containing the call labels
    OUT:
        3 rows x 1 column image where: 
        - top is spectrogram
        - middle is label
        - bottom is model prediction
    
    '''
    
    #plot spec
    plt.figure(figsize=(7, 12))
    plt.subplot(311)
    librosa.display.specshow(spec.T, x_axis='time' , y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    
    # plot LABEL
    plt.subplot(312)
    xaxis = range(0, np.flipud(label).shape[1]+1)
    yaxis = range(0, np.flipud(label).shape[0]+1)
    plt.yticks(np.arange(0.5, len(label_list)+0.5 ,1 ),label_list)
    plt.pcolormesh(xaxis, yaxis, np.flipud(label))
    plt.xlabel('time (s)')
    plt.ylabel('Calltype')
    plt.colorbar(label="Label")
    # plot_matrix(np.flipud(label), zunits='Label')
    
    plt.subplot(313)
    xaxis = range(0, np.flipud(pred.T).shape[1]+1)
    yaxis = range(0, np.flipud(pred.T).shape[0]+1)
    plt.yticks(np.arange(0.5, len(label_list)+0.5 ,1 ),label_list)
    plt.pcolormesh(xaxis, yaxis, np.flipud(pred.T))
    plt.xlabel('time (s)')
    plt.ylabel('Calltype')
    plt.colorbar(label="Probability")
    # plot_matrix(np.flipud(pred[0,:,:].T))
    
    plt.show()




