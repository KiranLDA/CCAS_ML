#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:31:48 2020

@author: kiran
"""

import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module

# from preprocess_functions import *
import postprocess.evaluation_metrics_functions as metrics
import pickle
import os
import numpy as np

# where to look for results and store them
results_path = '/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/metrics'

# Get the test files
test_log = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/label_table/testing_files_used.txt"

with open(test_log) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
testing_filenames = [x.strip() for x in content] 

label_dir = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/label_table"
pred_dir =  "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/pred_table"


#########################################################################
## Create thresholds per file
#########################################################################

# loop over each test file
for file_ID in testing_filenames: #file_ID = testing_filenames[0]
    label_ID = file_ID + "_LABEL_TABLE.txt"
    label_list = [os.path.join(label_dir,label_ID)]
    #loop over each threshold
    for high_thr in np.arange(0.3,0.9,0.1):     #high_thr = np.arange(0.3,0.9,0.1)[0]        
        high_thr = round(high_thr,1)
        pred_ID = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + ".txt"
        pred_list = [os.path.join(pred_dir, pred_ID)]
        evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
        Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
        
        # specify file names
        precision_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Precision.csv'
        recall_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Recall.csv'
        cat_frag_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Category_fragmentation.csv'
        time_frag_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Time_fragmentation.csv'
        confusion_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Confusion_matrix.csv'
        gt_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Label_indices.csv"
        pred_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Prection_indices.csv"
        match_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Matching_table.txt"
        timediff_filename = file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Time_difference.txt"
        
        # save files
        Prec.to_csv( os.path.join(results_path, precision_filename))
        Rec.to_csv( os.path.join(results_path, recall_filename))
        cat_frag.to_csv( os.path.join(results_path, cat_frag_filename))
        time_frag.to_csv(os.path.join(results_path, time_frag_filename))
        cf.to_csv(os.path.join(results_path, confusion_filename))
        gt_indices.to_csv(os.path.join(results_path, gt_filename ))
        pred_indices.to_csv(os.path.join(results_path, pred_filename ))
        with open(os.path.join(results_path, match_filename), "wb") as fp:   #Picklin
                  pickle.dump(match, fp)
        with open(os.path.join(results_path, timediff_filename), "wb") as fp:   #Pickling
            pickle.dump(offset, fp)    


#########################################################################
##  Create overall thresholds
#########################################################################

file_ID_list = [file_ID for file_ID in testing_filenames ]
label_list =  [os.path.join(label_dir,file_ID + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]
for high_thr in np.arange(0.3,0.9,0.1):     #high_thr = np.arange(0.3,0.9,0.1)[0]        
    high_thr = round(high_thr,1)    
    pred_list = [os.path.join(pred_dir,file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + ".txt" ) for file_ID in file_ID_list ]
    evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
    Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
    
    # specify file names
    precision_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Precision.csv'
    recall_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Recall.csv'
    cat_frag_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Category_fragmentation.csv'
    time_frag_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Time_fragmentation.csv'
    confusion_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Confusion_matrix.csv'
    gt_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Label_indices.csv"
    pred_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Prection_indices.csv"
    match_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Matching_table.txt"
    timediff_filename = "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + "_Time_difference.txt"    
    
    # save files
    Prec.to_csv( os.path.join(results_path, precision_filename))
    Rec.to_csv( os.path.join(results_path, recall_filename))
    cat_frag.to_csv( os.path.join(results_path, cat_frag_filename))
    time_frag.to_csv(os.path.join(results_path, time_frag_filename))
    cf.to_csv(os.path.join(results_path, confusion_filename))
    gt_indices.to_csv(os.path.join(results_path, gt_filename ))
    pred_indices.to_csv(os.path.join(results_path, pred_filename ))                  
    with open(os.path.join(results_path, match_filename), "wb") as fp:   #Picklin
              pickle.dump(match, fp)
    with open(os.path.join(results_path, timediff_filename), "wb") as fp:   #Pickling
        pickle.dump(offset, fp)    


