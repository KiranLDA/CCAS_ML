#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 15:02:10 2021

@author: kiran
"""
from sklearn.model_selection import KFold

class CrossValidator:
    
    def __init__(self, key_dict, batch_generator, model, 
                 n_folds = 10,
                 batch_size, epochs):
        
        # self.keys = keys # could be here for instance the individuals to loop over
        
        # key_dict # dictionary of keys that will be shuffled and cross-validated 
        
        kfold = KFold(n_folds, shuffle = True)
        
        # could pass examples in here?
        # pass in names of groups being shuffled, eg. per individual, year, day, 
        
        #
        all_keys = key_dict.keys()
        dummyvals = [idx for idx in range(len(all_keys))]
        
        for (train_idx, test_idx) in kfold.split(dummyvals, dummyvals):
            
            
        
        