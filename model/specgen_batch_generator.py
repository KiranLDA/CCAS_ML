#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:02:05 2020

@author: kiran
"""

import numpy as np
import tensorflow.keras as keras
import lib.audioframes
import lib.dftstream
import lib.endpointer
import preprocess.preprocess_functions as pre

class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 call_table_dict, #(callype, start, stop, filelocations)
                 mega_table, #(rename so it is clear it is label)
                 mega_noise_table,  #(rename so it is clear it is label)
                 spec_window_size,
                 # sample_size, #(method could be in data generator) 
                 window_size,
                 n_mels, window, fft_win , fft_hop , normalise,
                 label_for_noise,
                 label_for_other,
                 scaling_factor ,
                 n_per_call ,
                 other_ignored_in_training):
        
        self.call_table_dict = call_table_dict # (callype, start, stop, filelocations)
        self.mega_table # (rename so it is clear it is label)
        self.mega_noise_table = mega_noise_table # (rename so it is clear it is label)
        self.spec_window_size = spec_window_size
        self.label_for_noise = label_for_noise
        self.window_size = window_size
        self.n_mels = n_mels 
        self.window = window 
        self.fft_win = fft_win
        self.fft_hop = fft_hop
        self.normalise = normalise
        self.scaling_factor = scaling_factor
        self.n_per_call = n_per_call
        
        # Q do I need self here even if it is only used in init?
        # remove other from batch generations if necessary
        if other_ignored_in_training:
            del self.call_table_dict[label_for_other]
        
        #setup indexing given different calls have different sample sizes        
        self.indexes = dict()
        self.next_sample = dict()
        for calltype in call_table_dict.keys():
            # create shuffled indexes to randomise the sample order
            self.indexes[calltype] = np.arange(len(self.call_table_dict[calltype])) 
            np.random.shuffle(self.indexes[calltype]) 
            # create a variable which keeps track of samples
            self.next_sample[calltype] = 0
        
        # Q should this be here?
        self.mean_sample_size, self.sample_size = self.sampling_strategy() # Q is it correct to add self?
        self.batch_size = self.n_per_call * len(self.call_table_dict.keys())
        self.tot_batch_number = int(np.floor(self.mean_sample_size / self.batch_size))
        # Q shouldn't something about the total number of batches be here?
    
    def __len__(self):
        """len() - Number of batches in data"""
        return self.tot_batch_number
        
        
    def generate_example(self, calltype, call_num, to_augment):
        '''
        calltype, 
        call_num, 
        to_augment
        '''

        # extract the indexed call
        call = self.call_table_dict[calltype].iloc[(self.indexes[calltype][call_num])]
        
        # randomise the start a little so the new spectrogram will be a little different from the old
        # if the call is very long have a large range to draw the window
        if call["Duration"]>= self.spec_window_size:
            call_start = round(float(np.random.uniform(call["Start"]-self.spec_window_size/2, 
                                                       call["End"]-self.spec_window_size/2, 1)), 3)
        # if the call is short call, draw it from somewhere
        else:
            call_start = round(float(np.random.uniform((call["Start"]+call["End"])/2-self.spec_window_size, 
                                                       (call["Start"]+call["End"])/2)), 3)
        
        # load in a subsection of the spectrogram
        y, sr = librosa.load(call["wav_path"], sr=None, mono=False,
                             offset = call_start, duration =self.spec_window_size)
        
        call_stop = round(call_start + self.spec_window_size,3 )
        
        # convert from a time to a framenumber
        # start_lab = int(round(sr * decimal.Decimal(call_start),3))
        # stop_lab =  int(round(sr * decimal.Decimal(call_stop),3))            
        
        # have it as an array
        data_subset = np.asfortranarray(y)
        
        # If the call is to be augmented, do so, otherwise generate a spec and label from base data
        if to_augment :           
            
            # extract noise call
            noise_event = self.call_table_dict[self.label_for_noise].iloc[(self.indexes[self.label_for_noise][self.next_sample[self.label_for_noise]] )             
            # noise_event =  self.mega_noise_table.sample()
            
            #randomise the start and stop so the same section is never being used for the augmentation
            noise_start = round(float(np.random.uniform(noise_event.iloc[0]["Start"], 
                                                        (noise_event.iloc[0]["End"]-self.spec_window_size), 1)), 3)
            noise_stop = round(noise_start + spec_window_size,3 )
            y_noise, sr = librosa.load(noise_event["wav_path"], sr=None, mono=False,
                                 offset = noise_start, duration =self.spec_window_size)
            # start = int(round(sr * decimal.Decimal(noise_start),3))
            # stop =  int(round(sr * decimal.Decimal(noise_stop),3))
            noise_subset = np.asfortranarray(y_noise)
    
            # combine the two
            # Q randomise scaling factor (normal absolute dist 0.1-0.5)
            augmented_data = data_subset + noise_subset * self.scaling_factor
            # generate spectrogram
            augmented_spectrogram = pre.generate_mel_spectrogram(augmented_data, sr, 0, 
                                                             self.spec_window_size, 
                                                             self.n_mels, self.window, 
                                                             self.fft_win , self.fft_hop , self.normalise)
            # generate label
            augmented_label = pre.create_label_matrix(self.mega_label_table, augmented_spectrogram,
                                                  call_types, call_start, call_stop, 
                                                  self.label_for_noise)
        else:
            # generate spectrogram
            augmented_spectrogram = pre.generate_mel_spectrogram(data_subset, sr, 0, 
                                                             self.spec_window_size, 
                                                             self.n_mels, self.window, 
                                                             self.fft_win , self.fft_hop , self.normalise)
            # generate label
            augmented_label = pre.create_label_matrix(self.mega_label_table, augmented_spectrogram,
                                                  call_types, call_start, call_stop, 
                                                  self.label_for_noise)
        
        return augmented_spectrogram, augmented_label
    
    def __getitem__(self, batch_number):
        
        # Batch number is not yet defined anywhere but is part of user interfact
        
        # keep track of batch
        '''
        start_idx = batch_number * self.n_per_call
        stop_idx = (batch_number + 1) * self.n_per_call 
        '''
        # empty the batch at the beginning
        batch_label_data = []
        batch_spec_data = []
        
        # loop over every call   
        for calltype in self.call_table_dict:
            
            # Q is this right - I feel like it might impact parallelisation        
            
            # except if we have reached the end of the indexes, 
            # in which case they will need to be reshuffled
            if (self.next_sample[calltype] + self.n_per_call) > len(self.indexes[calltype]) :
                np.random.shuffle(self.indexes[calltype])  # reshuffle the data
                self.next_sample[calltype] = 0 #reset the index
            start_idx = self.next_sample[calltype] #readjust the start
            stop_idx = self.next_sample[calltype] + self.n_per_call 
            
            # move to the next sample for the next batch
            # self.next_sample[calltype] = stop_idx
            
            # for each call to be put in the batch generate an example            
            for call_num in range(start_idx, stop_idx):
                                # call_num = start_idx
                # calltype= "sn"
                
                # do a weighted coin flip
                augment = np.random.binomial(1, 
                                             float(sample_size.loc[sample_size["label"] == calltype, "prop_to_augment"]), 
                                             1)[0]
                
                # determine whether or not the call is to be augmented based on the coin flip
                to_augment = True if augment == 1 else False
                
                # map call number to actual call number                  
                # call_num = call_num % indexes[calltype].size
                
                # extract that call
                # call = self.call_table_dict[calltype].iloc[(indexes[calltype][call_num] ) 
                
                # generate the label and spectrogram
                label, spec = self.generate_example(calltype, call_num, to_augment) # Q is it correct to add self?
                
                # need to deal with noise index
                if calltype != self.label_for_noise and to_augment:
                    self.next_sample[self.label_for_noise] += 1 # Q is this ok?
                    if self.next_sample[self.label_for_noise]  > len(self.indexes[self.label_for_noise]) :
                        np.random.shuffle(self.indexes[self.label_for_noise])  # reshuffle the data
                        self.next_sample[self.label_for_noise] = 0 #reset the index
                
                # compile the batch
                batch_label_data = batch_label_data.append(label)
                batch_spec_data = batch_spec_data.append(spec)
                # add weight? decided no so that there are the same numbers of samples used in the training
        
                # Q need to deal with noise
                # move to the next sample for the next batch
                self.next_sample[calltype] += 1
       
        # # compute indecies /calls per batch calc (Batch size * call number)
        # mean_sample_size, sample_size = sampling_strategy()
        # sampling_strategy = sample_size["prop_to_augment"]
        # next_sample = call_num  # indexes[calltype][call_num]
        
        # Q is this correct that this needs to be returned
        return batch_label_data, batch_spec_data 

    def sampling_strategy(self):        
        # the purpose of this dictionary is to see how        
        sample_size = pd.DataFrame()
        
        for calltype in self.call_table_dict:
            sample_size = sample_size.append(pd.DataFrame([[calltype,
                                                            len(self.call_table_dict[calltype]),
                                                            sum(self.call_table_dict[calltype]["Duration"])]],
                                                          columns= ["label", "sample_size", "duration"]))
        mean_sample_size = round(sum(sample_size["sample_size"][sample_size["label"]!= self.label_for_noise])/len(sample_size["sample_size"][sample_size["label"]!= self.label_for_noise]))
        sample_size["calls_needed"] =  mean_sample_size - sample_size["sample_size"]
        sample_size["times_to_augment"] = sample_size["calls_needed"] /sample_size["sample_size"]
        sample_size["prop_to_augment"] = abs(sample_size["times_to_augment"]) / (abs(sample_size["times_to_augment"])+1)
        sample_size["samp_div_mean"]= sample_size["sample_size"] / mean_sample_size
        sample_size["size_with_augment"]= sample_size["sample_size"].copy()
        sample_size.loc[sample_size["size_with_augment"] <= mean_sample_size, "size_with_augment"] = mean_sample_size
        sample_size["prop_each_call"] = sample_size["size_with_augment"] / sum(sample_size["size_with_augment"])
        sample_size["equal_weights"] = 1/ len(sample_size["prop_each_call"]) 
        
        return mean_sample_size, sample_size
    
    
    # def plot_batch(self, batch_idx):
        
   
    def on_epoch_end(self):  
        """
        on_epoch_end - Bookkeeping at the end of the epoch
        :return:
        """
        self.epoch += 1  # Note next epoch
        if self.shuffle:
            for calltype in self.indexes.keys():
                np.random.shuffle(self.indexes[calltype])  # reshuffle the data
                self.next_sample[calltype] = 0
            

     
'''


import numpy as np
import tensorflow.keras as keras
import lib.audioframes
import lib.dftstream
import lib.endpointer

class DataGenerator(keras.utils.Sequence):
    def __init__(self, filenames, classes, batch_size=100, adv_ms=10, len_ms=20,
                  duration_ms = 300, shuffle=True, flatten=True, cache=True):
        """

        :param filenames: list of filenames containing examples
        :param classes:  class labels
        :param batch_size:  number items in batch
        :param adv_ms:  frame advance ms
        :param len_ms:  frame length ms
        :param duration_ms:  duration of speech to be classified
        :param shuffle:  Randomize order of examples each epoch?  True/False
        :param flatten:  Squash R^d features to R^1?  True/False
        :param cache:  Cache feature generation?  True/False (not for big data)
        """

        self.epoch = 0   # starting epoch
        self.N = len(filenames)
        self.filenames = filenames
        self.classes = np.array(classes)
        self.batch_size = batch_size
        # Store one-hot target encoding
        self.targets = keras.utils.to_categorical(classes)

        # framing parameters
        self.adv_ms = adv_ms
        self.len_ms = len_ms

        # speech length to extract
        self.duration_ms = duration_ms

        # Data will be handled in this order (changed if shuffle True)
        self.order = np.arange(self.N)

        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.order)

        self.flatten = flatten
        if cache:
            self.cache = dict()
        else:
            self.cache = None



    def __len__(self):
        """len() - Number of batches in data"""
        return int(np.floor(self.N / self.batch_size))

    def get_example(self, exampleidx):
        """
        get_example - Return features associated with a specific example
        :param exampleidx: Compute features for self.filenames[exampleidx]
        :param duration_ms:  How much data should be retained
        :return: tensor to be returned to model
        """

        file = self.filenames[exampleidx]

        if self.cache and file in self.cache:
            features = self.cache[file]  # already computed, use it
        else:
            # First time, extract spectra
            framer = lib.audioframes.AudioFrames(self.filenames[exampleidx],
                                                  self.adv_ms, self.len_ms)
            dftstream = lib.dftstream.DFTStream(framer)
            spectra = [s for s in dftstream]
            spectra = np.vstack(spectra)

            # Find which examples are likely to be speech
            ep = lib.endpointer.Endpointer(spectra)
            indices = ep.speech_indices(spectra)

            # Find midpoint of speech frames
            # indices are sorted, so middle one is median frame
            center_frame = indices[int(len(indices)/2)]

            # Number of frames to use on either side of the center frame
            # is half the duration divided by the frame rate
            frame_offset = int(self.duration_ms/(2*self.adv_ms))

            start_frame = center_frame - frame_offset
            end_frame = center_frame + frame_offset+1

            # Make sure we didn't go off either end and adjust if needed
            if start_frame < 0:
                start_frame = 0
                end_frame = 2 * frame_offset + 1
            elif end_frame > spectra.shape[0]:
                end_frame = spectra.shape[0]
                start_frame = end_frame - 2 * frame_offset - 1

            # Extract featurs likely to contain the word
            features = spectra[start_frame:end_frame, :]

            if self.flatten:
                features = features.flatten()

            if self.cache:
                self.cache[file] = features

        return features

    def get_labels(self):
        """
        get_labels() - Get category indices of all examples in set.
        If the data have been shuffled, categories are ordered according to
        the most recent shuffle and will no longer be valid after on_epoch_end()
        is called.
        :return:
        """
        if self.shuffle:
            labels = self.classes[self.order]
        else:
            labels = self.classes

        return labels



    def __getitem__(self, batch_idx):
        """
        Get idx'th batch
        :param batch_idx: batch number
        :return: (examples, targets) - Returns tuple of data and targets for
            specified batch
        """

        # Determine indices to process
        start = batch_idx * self.batch_size
        stop = start + self.batch_size  # one past last

        examples = []
        for idx in range(start, stop):
            data = self.get_example(self.order[idx])
            examples.append(data)

        examples_tensor = np.stack(examples, axis=0)

        targets = self.targets[self.order[start:stop],:]

        return examples_tensor, targets



    def on_epoch_end(self):
        """
        on_epoch_end - Bookkeeping at the end of the epoch
        :return:
        """
        self.epoch += 1  # Note next epoch
        if self.shuffle:
            np.random.shuffle(self.order)  # reshuffle the data

'''