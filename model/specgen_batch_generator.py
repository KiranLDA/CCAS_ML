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

generator.py
Displaying generator.py.