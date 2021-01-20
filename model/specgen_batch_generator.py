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


class DataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 call_table_dict, #(callype, start, stop, filelocations)
                 mega_table, 
                 mega_noise_table,  #(rename so it is clear it is label)
                 spec_window_size,
                 sample_size, #(method could be in data generator) 
                 window_size,
                 fft_params, n_mels, window, fft_win , fft_hop , normalise,
                 label_for_noise,
                 scaling_factor #,
                 # n_per_call = 3
                 ):
        
        self.call_table_dict = call_table_dict # (callype, start, stop, filelocations)
        self.mega_table # (rename so it is clear it is label)
        self.mega_noise_table = mega_noise_table # (rename so it is clear it is label)
        self.spec_window_size = spec_window_size
        self.sample_size = sample_size # (method could be in data generator) 
        self.window_size = window_size
        self.fft_params = fft_params
        self.scaling_factor = scaling_factor
        self.label_for_noise = label_for_noise
        # self.n_per_call = n_per_call
        
        self.indexes = dict()
        for call_key in call_table_dict.keys():
            self.indexes[call_key] = np.arange(len(self.call_table_dict[call_key])) 
            np.random.shuffle(self.indexes[call_key]) 
        
        
        
    def generate_example(self,
                         call):
        '''
        call_number and call_type      
        Load signal
        dice roll for augmentation
        if noise_augmentation:
            load noise
            add noise (*scaling factor -> uniform distribution) to signal
        generate spec
        Generate label
        '''
        imports: calltype, index, call
        # Get the call
        
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
        
        #convert from a time to a frame
        start_lab = int(round(sr * decimal.Decimal(call_start),3))
        stop_lab =  int(round(sr * decimal.Decimal(call_stop),3))            
        
        # suset the wav
        data_subset = np.asfortranarray(y)
        
        
        # Get the noise
        
        # randomly choose a noise file - same as above, only with noise
        noise_event =  self.mega_noise_table.sample()#random.choice([x for x in noise_filepaths if label_for_noise in x])#glob.glob(folder + "/*" + calltype +".npy")

       
        noise_start = round(float(np.random.uniform(noise_event.iloc[0]["Start"], 
                                                    (noise_event.iloc[0]["End"]-self.spec_window_size), 1)), 3)
        noise_stop = round(noise_start + spec_window_size,3 )
        y_noise, sr = librosa.load(noise_event["wav_path"], sr=None, mono=False,
                             offset = noise_start, duration =self.spec_window_size)
        # start = int(round(sr * decimal.Decimal(noise_start),3))
        # stop =  int(round(sr * decimal.Decimal(noise_stop),3))
        noise_subset = np.asfortranarray(y_noise)

        # combine the two
        augmented_data = data_subset + noise_subset * self.scaling_factor
        # generate spectrogram
        augmented_spectrogram = generate_mel_spectrogram(augmented_data, sr, 0, 
                                                         self.spec_window_size, 
                                                         self.n_mels, self.window, 
                                                         self.fft_win , self.fft_hop , self.normalise)
        # generate label
        augmented_label = create_label_matrix(self.mega_label_table, augmented_spectrogram,
                                              call_types, call_start, call_stop, 
                                              self.label_for_noise)
        
        return augmented_spectrogram, augmented_label
    
    def __getitem__(self, batch_number):
        # numb of each type
        # tot num of calls gend
        # need to init batch_data
        
        # keep track of batch
        start_idx = batch_number * self.n_per_call
        stop_idx = (batch_number + 1) * self.n_per_call   
        
        # for every call compile a batch        
        for calltype in call_dict:
            for call_num in range(start_idx, stop_idx):
                
                
                
                
                # map call number to actual call number                  
                call_num = call_num % indexes[calltype].size
                
                # extract that call
                call = call_table_dict[calltype].iloc[(indexes[calltype][call_num] ) ]
                
                # generate the label and spectrogram
                label, spec = generate_example(call, calltype,call_num)
                
                # compile the batch
                batch_label_data = batch_label_data.append(label)
                batch_spec_data = batch_spec_data.append(spec)
                # add weight? 
        
        # compute indecies /calls per batch calc (Batch size * call number)
        mean_sample_size, sample_size = sampling_strategy()
        sampling_strategy = sample_size["prop_to_augment"]
        next_sample = call_num  # indexes[calltype][call_num]
        
        
        # start, end = computed batch index
        
        # for idx in range(start, end):
            # pick a call type based sampling strategy  
            # call_idx = next_sample[call type]
            # get label spectrogram for call type & idx
            # # option 1, only reshuffle at batch end, we use moduluo operator % to wrap around
            # next_sample[call type] = next_sample[call type] % #of call type
            # # option 2, reshuffle at end of samples
            # if next_sample[call type] + 1 > # samples
            # reshuffle
            # else add 1

    def sampling_strategy(self):        
        # the purpose of this dictionary is to see how        
        sample_size = pd.DataFrame()
        
        for label in self.call_table_dict:
            sample_size = sample_size.append(pd.DataFrame([[label,
                                                            len(self.call_table_dict[label]),
                                                            sum(self.call_table_dict[label]["Duration"])]],
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
   
     def on_epoch_end(self):  
        """
        on_epoch_end - Bookkeeping at the end of the epoch
        :return:
        """
        self.epoch += 1  # Note next epoch
        if self.shuffle:
            for calltype in self.indexes.keys():
                np.random.shuffle(self.indexes[calltype])  # reshuffle the data
            
            
#---------------------------------------------------------
#   OUTLINE
#---------------------------------------------------------

          
# # inputs:
#     call_table_dict (callype, start, stop, filelocations)
#     mega_table /mega_noise_table  (rename so it is clear it is label)
#     sample_size (method could be in data generator)    
#     window_size
#     fft_params
    
# # Step 1:
#     __init__
#         epoch (more needed because epochs are smaller)
#         cache (open file handles)
#         indexing (map batches to index numbers) Batch size * dataset length
# # Step 2:
#     generate_example
#         Load signal
#         dice roll for augmentation
#         if noise_augmentation:
#             load noise
#             add noise (*scaling factor -> uniform distribution) to signal
#         generate spec
#         Generate label
        
# # Step 3:
#     __getitem__
#         compiling examples into batch (append)
#         compute indecies /calls per batch calc (Batch size * call number)
    
# # Step 4:
#     on_epoch_end
#         shuffling / keeping track of indexes

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