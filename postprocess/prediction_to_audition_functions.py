#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:17:12 2020

@author: kiran
"""
import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")


import preprocess.preprocess_functions as pre
import pandas as pd
import os
import postprocess.merge_predictions_functions as post


#pandas parameters for reading in predictions tables
sep=";"
engine=None

# location of the raw data
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

# location of the processed data i.e. the root directory for saving spectrograms and labels
save_data_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'

# Note that the lines below don't need to be modified 
# unless you have/want a different file structure
# They will create a specific file sub directory pattern in save_data_path

# 1. where to save the label tables
# save_label_table_path = os.path.join(save_data_path, 'label_table')
# save_pred_table_path = os.path.join(save_data_path, 'prediction_table')

# 2. where to save the training specs and labels
train_path = os.path.join(save_data_path,'train_data')
save_spec_train_path = os.path.join(train_path , "spectrograms")
save_mat_train_path = os.path.join(train_path , "label_matrix")
save_label_table_train_path = os.path.join(train_path, 'label_table')


# 3. where to save the testing labels and specs and predictions
test_path = os.path.join(save_data_path, 'test_data')
save_spec_test_path = os.path.join(test_path , "spectrograms")
save_mat_test_path = os.path.join(test_path , "label_matrix")
save_pred_test_path = os.path.join(test_path , "predictions")
save_label_table_test_path = os.path.join(test_path, 'label_table')
save_pred_table_test_path = os.path.join(test_path, 'pred_table')


save_pred_audition_path = os.path.join(test_path, 'pred_audition_table')


#if those directories do not exist, create them
directories = [save_data_path, 
               train_path, save_spec_train_path, save_mat_train_path, save_label_table_train_path,
               test_path, save_spec_test_path, save_mat_test_path, save_label_table_test_path, save_pred_test_path, 
               save_pred_table_test_path, save_pred_audition_path]

for diri in directories:
    if not os.path.exists(diri):
        os.mkdir(diri)
      




prediction_table_paths = os.listdir(save_pred_table_test_path)


for path in prediction_table_paths:
    #for testing
    # path=prediction_table_paths[5]
    '''
    # read in the 
    pred_table = pd.read_csv(os.path.join(save_pred_table_test_path, path), sep=sep, header=0, engine = engine) 
    pred_table.rename(columns={'Label':'Name'}, inplace=True)
    
    audition_table = pred_table[["Name", "Start"]]
    
    f = lambda x: pre.convert_secs_to_fulltime(x["Start"]) 
    audition_table["Start"] = pred_table.apply(f, axis=1)
    
    f = lambda x: pre.convert_secs_to_fulltime(x["Duration"])    
    audition_table["Duration"] = pred_table.apply(f, axis=1)
    audition_table["Time Format"] = "decimal"
    
    audition_table["Type"] = "Cue"
    audition_table["Description"] = ""
    '''
    audition_table = post.pred_to_audition(os.path.join(save_pred_table_test_path, path), 
                                      sep= ";", engine = None)
    
    filename = path.split(".txt")[0]
    filename = filename + "_AUDITION.csv"
    
    audition_table.to_csv(os.path.join(save_pred_audition_path, filename), header=True, index=None, sep='\t', mode='a') 
    
