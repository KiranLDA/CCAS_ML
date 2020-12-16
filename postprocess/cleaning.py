#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 11:03:50 2020

@author: kiran
"""
import pandas as pd



def compile_pred_table(detections, fft_hop, fromi, toi, loop_table, loopi, call_types):

    
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
    
    return pred_table
    

