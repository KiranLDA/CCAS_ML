#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:04:18 2020

@author: kiran
"""
import numpy as np
from random import shuffle

class Batch_Generator():
    
    ''' 
    spectro_list: 
        list of spectrograms to create batches with (full path required)
    label_list:   
        list of label matrices to create batches with (full path required)
    batch: 
        batch size (e.g. 32, 64)
    shuffling: 
        True or False, whether or not to shuffle the training files between epochs
    '''
    
    def __init__(self, spectro_list, label_list, batch, shuffling):
        self.spectro_list = spectro_list
        self.batch = batch
        self.label_list = label_list
        self.shuffling = shuffling
        self.idx = 0 
    
    def __iter__(self):
        self.idx = 0
        return self    
    
    def __next__(self):
        spectros = []
        labels = []
        if self.idx >= ((len(self.spectro_list)//self.batch) * self.batch) :
            self.idx = 0
        if self.shuffling == True & self.idx == 0:
            c = list(range(len(self.spectro_list)))
            shuffle(c)
            self.spectro_list = [self.spectro_list[i] for i in c]
            self.label_list = [self.label_list[i] for i in c]#['_MAT_'.join(x.split("_SPEC_")) for x in self.spectro_list]
        #     # c = list(zip(self.spectro_list, self.label_list))
        #     # shuffle(c)
        #     # self.spectro_list, self.label_list = zip(*c)
        # else:
        print(" iteration: " , self.idx)
        last = min(self.idx + self.batch, len(self.spectro_list))  

        for i in range(self.idx, last):
            spectros.append(np.load(self.spectro_list[i]).T)
            labels.append(np.load(self.label_list[i]).T) 

        self.idx = last
        # spectros = np.expand_dims(np.asarray(spectros),4)
        spectros = np.asarray(spectros)
        spectros =  spectros[..., np.newaxis]
        labels = np.asarray(labels)
        return spectros, labels
    
    def steps_per_epoch(self):
        steps = int(len(self.spectro_list)/self.batch)
        return steps
    
    
'''
spectros = []
labels = []
      
if self.idx >= self.steps_per_epoch() * self.batch :
    self.idx = 0
    if ((self.idx == 0) & (self.shuffling)):
        c = list(range(len(self.spectro_list)))
        shuffle(c)
        self.spectro_list = [self.spectro_list[i] for i in c]
        self.label_list = ['_MAT_'.join(x.split("_SPEC_")) for x in self.spectro_list]
# else:
print(" iteration: " , self.idx)
last = min(self.idx + self.batch, len(self.spectro_list))  
for i in range(self.idx, last):
    spectros.append(np.load(self.spectro_list[i]).T)
    labels.append(np.load(self.label_list[i]).T)

self.idx = last
# spectros = np.expand_dims(np.asarray(spectros),4)
spectros = np.asarray(spectros)
spectros =  spectros[..., np.newaxis]
labels = np.asarray(labels)

return spectros, labels
'''
    
class Weighted_Batch_Generator():
    
    ''' 
    spectro_list: 
        list of spectrograms to create batches with (full path required)
    label_list:   
        list of label matrices to create batches with (full path required)
    batch: 
        batch size (e.g. 32, 64)
    shuffling: 
        True or False, whether or not to shuffle the training files between epochs
    '''
    
    def __init__(self, spectro_list, label_list, weight_list, batch, shuffling):
        self.spectro_list = spectro_list
        self.batch = batch
        self.label_list = label_list
        self.weight_list = weight_list
        self.shuffling = shuffling
        self.idx = 0 
    
    def __iter__(self):
        self.idx = 0
        return self    
    
    def __next__(self):
        spectros = []
        labels = []
        weights =[]
        if self.idx >= ((len(self.spectro_list)//self.batch) * self.batch) :
            self.idx = 0
        if self.shuffling == True & self.idx == 0:
            c = list(range(len(self.spectro_list)))
            shuffle(c)
            self.spectro_list = [self.spectro_list[i] for i in c]
            self.label_list = [self.label_list[i] for i in c]
            self.weight_list = [self.weight_list[i] for i in c]
            #['_MAT_'.join(x.split("_SPEC_")) for x in self.spectro_list]
        #     # c = list(zip(self.spectro_list, self.label_list))
        #     # shuffle(c)
        #     # self.spectro_list, self.label_list = zip(*c)
        # else:
        print(" iteration: " , self.idx)
        last = min(self.idx + self.batch, len(self.spectro_list))  

        for i in range(self.idx, last):
            spectros.append(np.load(self.spectro_list[i]).T)
            labels.append(np.load(self.label_list[i]).T) 
            weights.append(self.weight_list[i])

        self.idx = last
        # spectros = np.expand_dims(np.asarray(spectros),4)
        spectros = np.asarray(spectros)
        spectros =  spectros[..., np.newaxis]
        labels = np.asarray(labels)
        weights = np.asarray(weights)
        return spectros, labels, weights
    
    def steps_per_epoch(self):
        steps = int(len(self.spectro_list)/self.batch)
        return steps
    
      

class Test_Batch_Generator():
    
    '''this generatoes only batches of spectrograms'''
    
    def __init__(self, spectro_list, label_list, batch):
        self.spectro_list = spectro_list
        self.batch = batch
        self.label_list = label_list
        self.idx = 0
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        spectros = []
        print(" iteration: " , self.idx)
        
        start = self.idx
        stop = min(start + self.batch, len(self.spectro_list) )
        # if stop < len(self.spectro_list) :
        for i in range(start, stop):
            spectros.append(np.load(self.spectro_list[i]).T)
        spectros = np.asarray(spectros)
        spectros =  spectros[..., np.newaxis]      
        self.idx = stop #- 1 #+ 1
        
        return spectros 
    
    
        #------
        # spectros = []
        # print(" iteration: " , self.idx)
        # for i in range(self.idx, self.idx + self.batch):
        #     spectros.append(np.load(self.spectro_list[i]).T)
        # self.idx = self.idx + self.batch#last
        # if self.idx >= self.steps_per_epoch() * self.batch:
        #     self.idx = 0
        # spectros = np.asarray(spectros)
        # spectros =  spectros[..., np.newaxis]   
        # # spectros = np.expand_dims(np.asarray(spectros),4)
        # return spectros 
    
    def steps_per_epoch(self):
        steps = int(len(self.spectro_list)/self.batch)
        return steps



class Train_Batch_Generator():
    
    ''' this generates batches of both spectrograms and labels'''
    
    def __init__(self, spectro_list, label_list, batch):
        self.spectro_list = spectro_list
        self.batch = batch
        self.label_list = label_list
        self.idx = 0
    
    def __iter__(self):
        self.idx = 0
        return self
    
    def __next__(self):
        spectros = []
        labels =[]
        print(" iteration: " , self.idx)
        start = self.idx
        stop = min(start + self.batch, len(self.spectro_list) )
        if self.idx >= self.steps_per_epoch() * self.batch :
            self.idx = 0
        for i in range(start, stop):
            spectros.append(np.load(self.spectro_list[i]).T)
            labels.append(np.load(self.label_list[i]).T)
        spectros = np.asarray(spectros)
        spectros =  spectros[..., np.newaxis]      
        labels = np.asarray(labels)
        self.idx = stop #- 1 #+ 1
        return spectros, labels 
    
    def steps_per_epoch(self):
        steps = int(len(self.spectro_list)/self.batch)
        return steps

