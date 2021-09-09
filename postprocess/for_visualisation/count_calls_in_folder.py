#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:21:28 2020

this code is setup to look in a directory and find out how many calls of a particular type there are

@author: kiran
"""
#import libraries
import os

call_types = {
    'cc' :["cc","Marker", "Marque"],
    'sn' :["sn","subm", "short","^s$", "s "],
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL", "so "],
    'al' :["al","ALARM"],
    'beep':["beep"],
    'synch':["sync"],
    'oth':["oth","other","lc", "lost",
           "hyb","HYBRID","fu","sq","\+",
           "ukn","unknown",          
           "x",
           "\%","\*","\#","\?","\$"
           ],
    'noise':['start','stop','end','skip']
    }

calls = list(call_types.keys())
diri = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/train_data/label_matrix"


files = os.listdir(diri)

call_count = dict()
for calli in calls:
    call_count[calli] = len([i for i in files if calli in i] )
    
