#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:15:51 2020

@author: kiran
"""

import pandas as pd
# for every training file get the label table

mega_table = pd.DataFrame()


# Start the loop by going over every single labelled file id
for file_ID in training_filenames:
    # file_ID = label_filenames[2]
    
    # save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
    
    # only generate training files if they don't already exist    
    if not os.path.exists(os.path.join(save_label_table_train_path, save_label_table_filename)):
            
        # find the matching audio for the label data
        audio_path = [s for s in audio_filepaths if file_ID in s][0]
        
        # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
        label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
        
        print ("File being processed : " + label_path)    
        
        # create a standardised table which contains all the labels of that file - also can be used for validation
        label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, 
                                       label_for_other, label_for_noise, engine, True)
        # replace duration of beeps with 0.04 seconds - meerkat particularity
        label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
        label_table.loc[label_table["beep"] == True, "End"] += 0.04
    
        # Don't run the code if that file has already been processed
        # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
        #     continue
        # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
        label_table.to_csv(os.path.join(save_label_table_train_path, save_label_table_filename), header=True, index=None, sep=';')
        
        #save the label tables with other, but for the purpose of labelling, remove other
        if other_ignored_in_training:
            label_table = label_table[label_table[label_for_other] == False]
            label_table= label_table.reset_index(drop=True)
        
        mega_table = pd.concat([mega_table, label_tabletable])
        
# Once all the label tables are generated, combine them all into one big list of calls
# generate list of noise 