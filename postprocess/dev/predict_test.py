#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:44:59 2020

@author: kiran
"""


import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer/")

import numpy as np
from model.batch_generator import Test_Batch_Generator
import keras.models
import os


#specify the locations of everything
model_name = "savedmodel.h5"
model_location = "/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_test_2020-06-22_17:31:25.589743"#"/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_test_2020-06-15_17:40:52.199581/"
test_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/subset_4_testing/test_data/'
save_spec_path = os.path.join(test_path + "spectrograms")
save_mat_path = os.path.join(test_path + "label_matrix")


#-----------------------------------------------------
#load the model
RNN_model = keras.models.load_model(os.path.join(model_location, model_name))
 
#-----------------------------------------------------
# create a list of all the files
x_test_files = os.listdir(save_spec_path)
y_test_files = os.listdir(save_mat_path) 

#append full path to test files
x_test_files = [os.path.join(save_spec_path, x) for x in x_test_files ]
y_test_files = [os.path.join(save_mat_path, x) for x in y_test_files ]

#-----------------------------------------------------
# Predict over a batch

#only need one epoch anyway
batch = 64


#get the test files into the data generator
test_generator = Test_Batch_Generator(x_test_files , y_test_files, batch)

#predict
predictions = RNN_model.predict_generator(test_generator, test_generator.steps_per_epoch() )




#-----------------------------------------------------
# Predict for a radom file
import postprocess.visualise_prediction_functions as pp

#   Find a random spectrogram of a particular call type
spec, label, pred = pp.predict_label_from_random_spectrogram(RNN_model, "ag", save_spec_path, save_mat_path)

# and plot it
pp.plot_spec_label_pred(spec, label, pred)






