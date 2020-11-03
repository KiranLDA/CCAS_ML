#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:54:48 2020

@author: kiran
"""




import pandas as pd

def create_noise_table(label_table, label_for_noise, label_for_startstop):
    noise_table = pd.DataFrame()
    end = label_table.loc[0:(len(label_table["End"])-2), "End"].reset_index()
    start = label_table.loc[1:(len(label_table["End"])-1), "Start"].reset_index()
    label_start = label_table.loc[0:(len(label_table["End"])-2), "Label"].reset_index()
    label_end = label_table.loc[1:(len(label_table["End"])-1), "Label"].reset_index()
    noise_table["Label"] =  "Noise"#label["Label"]
    noise_table["Start"] = end["End"]
    noise_table["Duration"] = start["Start"]-end["End"]
    noise_table["End"] = start["Start"]
    
    noise_table["remove"] = False
    for labeli in label_for_startstop:
        noise_table.loc[label_start["Label"].str.contains(labeli, regex=True, case = False), 
                        "remove"] = True
        noise_table.loc[label_end["Label"].str.contains(labeli, regex=True, case = False), 
                        "remove"] = True
    noise_table = noise_table.loc[ noise_table["remove"]== False, ["Label", "Start", "Duration", "End"]] 
    noise_table["Label"] = "Noise"
    for col in set(label_table.columns) - set(noise_table.columns ):
        noise_table[col] = False
    noise_table[label_for_noise] = True