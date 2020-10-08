#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:59:09 2020

@author: kiran
"""


import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

testfile = '/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/metrics/HM_HMB_R11_AUDIO_file_5_(2017_08_24-06_44_59)_ASWMUX221163_PRED_TABLE_thr_0.2-0.3Confusion_matrix.csv'
with open(testfile, newline='') as csvfile:
    array = list(csv.reader(csvfile))

print(array)




# array = [[13,1,1,0,2,0],
#          [3,9,6,0,1,0],
#          [0,0,16,2,0,0],
#          [0,0,0,13,0,0],
#          [0,0,0,0,15,0],
#          [0,0,1,0,0,15]]

df_cm = pd.DataFrame(array)#, range(6), range(6))


new_header = df_cm.iloc[0] #grab the first row for the header
df_cm = df_cm[1:] #take the data less the header row
df_cm.columns = new_header #set the header row as the df header

new_header = df_cm['']
df_cm = df_cm.drop('', 1)
df_cm.index = new_header
df_cm.index.name= None
df_cm.columns.name= None
# df_cm = df_cm.set_index('')


df_cm = df_cm.astype(float)
# pd.DataFrame(data=np.arange(10),columns=['v']).astype(float)
# print df

# plt.figure(figsize=(10,7))
sn.set(font_scale=1) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()



##############################################################
# Plot confusion matrices for all levels
##############################################################

results_path = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/metrics/"
normalise = True

file_ID_list = [file_ID for file_ID in testing_filenames ]
label_list =  [os.path.join(label_dir,file_ID + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]
for high_thr in np.arange(0.3,0.9,0.1):     #high_thr = np.arange(0.3,0.9,0.1)[0]        
    high_thr = round(high_thr,1)
    confusion_filename = os.path.join(results_path, "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Confusion_matrix.csv')
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
    
    #move last negatives to end
    col_name = "FN"
    last_col = df_cm.pop(col_name)
    df_cm.insert(df_cm.shape[1], col_name, last_col)
    
    # remove noise
    df_cm = df_cm.drop("noise", axis=1)
    df_cm = df_cm.drop("noise", axis=0)
   
   
    #make sure it is a float
    df_cm = df_cm.astype(float)
    
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
    ax.set_title(str(high_thr) )
    plt.show()




##############################################################
# Plot individual variability per call type
##############################################################




import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module


# from preprocess_functions import *
import postprocess.evaluation_metrics_functions as metrics
import pickle
import os
import numpy as np
import pandas as pd
results_path = '/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/metrics'

# Get the test files
test_log = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/label_table/testing_files_used.txt"

with open(test_log) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
testing_filenames = [x.strip() for x in content] 

label_dir = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/label_table"
pred_dir =  "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/pred_table"

high_thr = 0.3
high_thr = round(high_thr,1)


# Recall_filelist = [os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Precision.csv') for file_ID in testing_filenames ]
# Precision_filelist = [os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Recall.csv') for file_ID in testing_filenames ]

precision = pd.DataFrame()
recall = pd.DataFrame()



for file_ID in testing_filenames:
    #file_ID = testing_filenames[0]
    precision_row = pd.read_csv(os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Precision.csv'))
    precision_row.rename(columns={"Unnamed: 0":'File'}, inplace=True)
    precision_row["File"] = file_ID
    precision_row.index.name= None
    precision = pd.concat([precision, precision_row]) 

    recall_row = pd.read_csv(os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Recall.csv'))
    recall_row.rename(columns={"Unnamed: 0":'File'}, inplace=True)
    recall_row["File"] = file_ID
    recall_row.index.name= None
    recall = pd.concat([recall, recall_row]) 
    
recall.hist(column="cc")
recall.hist(column="sn")
recall.hist(column="al")
recall.hist(column="soc")
recall.hist(column="mo")
recall.hist(column="agg")



for calltype in ["cc","sn","al","soc","mo","agg"]:
    x = recall[calltype].values
    y = precision[calltype].values
    n = precision["File"]
    n =  [f.split("_")[3][0:4] for f in n] # ["".join(f.split("_")[3])for f in n] 
    
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(calltype + " threshold:" + str(high_thr) ) 
    plt.xlabel('Recall') 
    plt.ylabel('Precision') 
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

# plot individual precision recall for only lead call

for high_thr in [0.3,0.9 ]:
    high_thr = round(high_thr,1)
    for file_ID in testing_filenames:
        #file_ID = testing_filenames[0]
        precision_row = pd.read_csv(os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Precision.csv'))
        precision_row.rename(columns={"Unnamed: 0":'File'}, inplace=True)
        precision_row["File"] = file_ID
        precision_row.index.name= None
        precision = pd.concat([precision, precision_row]) 
    
        recall_row = pd.read_csv(os.path.join(results_path, file_ID + "_PRED_TABLE_thr_0.2-" + str(high_thr) + 'Recall.csv'))
        recall_row.rename(columns={"Unnamed: 0":'File'}, inplace=True)
        recall_row["File"] = file_ID
        recall_row.index.name= None
        recall = pd.concat([recall, recall_row]) 
    for calltype in ["ld"]:
        x = recall[calltype].values
        y = precision[calltype].values
        n = precision["File"]
        n =  [f.split("_")[3][0:4] for f in n] # ["".join(f.split("_")[3])for f in n] 
        
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(calltype + " threshold:" + str(high_thr) ) 
        plt.xlabel('Recall') 
        plt.ylabel('Precision') 
        
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))


# plot overall precision and recall
precision = pd.DataFrame()
recall = pd.DataFrame()

for high_thr in np.arange(0.3,0.9,0.1):
    high_thr = round(high_thr,1)
    # os.path.join(results_path, "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Confusion_matrix.csv')
    #file_ID = testing_filenames[0]
    precision_row = pd.read_csv(os.path.join(results_path, "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Precision.csv'))
    precision_row.rename(columns={"Unnamed: 0":'Threshold'}, inplace=True)
    precision_row["Threshold"] = high_thr 
    precision_row.index.name= None
    precision = pd.concat([precision, precision_row]) 

    recall_row = pd.read_csv(os.path.join(results_path, "Overall_PRED_TABLE_thr_0.2-" + str(high_thr) + '_Recall.csv'))
    recall_row.rename(columns={"Unnamed: 0":'Threshold'}, inplace=True)
    recall_row["Threshold"] = high_thr 
    recall_row.index.name= None
    recall = pd.concat([recall, recall_row]) 
    
for calltype in ["cc","sn","al","soc","mo","agg","ld"]:
    x = recall[calltype].values
    y = precision[calltype].values
    n = precision["Threshold"]
    # n =  [f.split("_")[3][0:4] for f in n] # ["".join(f.split("_")[3])for f in n] 
    
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(calltype) 
    plt.xlabel('Recall') 
    plt.ylabel('Precision') 
    
    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
