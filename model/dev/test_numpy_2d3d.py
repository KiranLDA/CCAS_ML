#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:22:53 2020

@author: kiran
"""

import numpy as np
spectro_list = x_train_filelist
batch = 32
label_list = y_train_filelist
shuffling = False
idx = 0 


spectros = []
labels = []
if idx >= ((len(spectro_list)//batch) * batch) :
    idx = 0
if shuffling == True & idx == 0:
    c = list(range(len(spectro_list)))
    shuffle(c)
    spectro_list = [spectro_list[i] for i in c]
    label_list = [label_list[i] for i in c]

print(" iteration: " , idx)
last = min(idx + batch, len(spectro_list))  

for i in range(idx, last):
    print(i)
    spectros.append(np.load(spectro_list[i]).T)
    labels.append(np.load(label_list[i]).T) 

np.atleast_3d(np.load(spectro_list[i]).T).shape






(np.load(spectro_list[i]).T).shape

np.concatenate(spectros,axis=1).shape
# i=1
# spectros.append(np.load(spectro_list[i]).T)
# labels.append(np.load(label_list[i]).T) 
spectros = np.array(spectros)
spectros = np.concatenate( spectros, axis=0 )
spectros.shape
# spectros = np.expand_dims(np.asarray(spectros),4)
spectros = np.asarray(spectros)
spectros.shape
spectros =  spectros[..., np.newaxis]
spectros.shape
labels = np.asarray(labels)
labels.shape


idx = last



return spectros, labels
