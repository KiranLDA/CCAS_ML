#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:37:26 2021

@author: kiran
"""


"""
postprocess - Module for converting frame-level probabilities to labels
"""

# Standard modules
import datetime
from collections import namedtuple

# Add-on Python modules
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import preprocess.preprocess_functions as pre


TupPrevious = namedtuple("previous", ("score_idx", "frame_idx",))
TupDetection = namedtuple("detection", ('category', 'start', 'end', 'scores'))


def merge_p(probabilities, labels, starttime=datetime.timedelta(0), frameadv_s=0.010, specadv_s=3.0,
            low_thr=.3, high_thr=.4, debug=0):
    """
    Given a list of probability matrices, merge the matrices and find calls.
    :param probabilities:   List of score matrices, each N frames by C classes.  It is assumed
       that probabiliites[:,-1] is a noise class, and that matrices follow one another with possible
       overlap.
    :param starttime:  start of first score matrix
    :param frameadv_s: Time between consecutive frames in seconds
    :param specadv_s: Time between start of one score matrix and the next in seconds
    :param low_thr:  low-water mark (see high_thr)
    :param high_thr: high-water mark.  Detections are triggered when they pass the minimum of the
       high-water mark.  We expand the detection to either side until the score drops below the low-water
       mark.
    :param debug: integer indicating debug level.
      0 - no information
      5 - shows detections for each processing block
    :return:
       List of TupDetection named tuples sorted by start time
       Fields:  category, start, end, scores,
    """

    # Get number of score matrices, number of categories and frames per score matrix (assume uniform)
    N = len(probabilities)
    (framesN, catsN)= probabilities[0].shape

    specadv_n = int(specadv_s / frameadv_s)

    current = starttime

    more = N > 0  # frames to process?
    score_idx = 0
    previous_scores = []  # score matrices that are not yet complete (none now)

    detections = []  # List of detection tuples
    scanners = [FindDetections(labels[l], low_thr, high_thr, specadv_n) for l in range(catsN-1)]

    while more:
        if (score_idx % 1000) == 0:
            print("Detecting calls in probability streams:  %d / %d"%(score_idx, N))

        # previous_scores contains the list of named tuples (TupPrevious) that
        # describe previous score matrices that have not been fully processed.
        # While processing previous_scores, we will be building up a new list
        # for the next pass that includes the current frame being processed.

        previous_updated = []
        if (score_idx < N):
            # Processing the next frame, grab scores up to where next frame starts
            prob = probabilities[score_idx][0:specadv_n, :]
            # First time we processed this frame, so add to the list that will
            # be examined in the next iteration of the algorithm.
            previous_updated.append(TupPrevious(score_idx = score_idx,
                                                frame_idx = specadv_n))
        else:
            # No more frames, create zeros and fill in based on previous frames
            prob = np.zeros((specadv_n, probabilities[-1].shape[-1]))


        for s in list(previous_scores):
            # Find best score in overlaps and advance within overlapped score frame.
            # If there might be more overlaps, update and add to previous_updated
            prev = probabilities[s.score_idx]
            start = s.frame_idx
            stop = start + specadv_n
            if stop <= framesN:
                """
                Ignoring the noise class, find out which likelihood is highest and keep it.
                For now, we're taking the best class, but we might want the sum of the non-noise
                classes due to the cases with multiple classes (calls) being active
                """
                # Access the likelihoods that overlap with the current segment of the current score set
                prob_old = probabilities[s.score_idx][start:stop]

                # Next frame advance processed might need to look at the next step in this frame
                if stop + specadv_n <= framesN:
                    previous_updated.append(TupPrevious(score_idx = s.score_idx, frame_idx=stop))

                # Get best class scores
                best = np.max(prob[:,:-1], axis=1)
                best_old = np.max(prob_old[:,:-1], axis=1)
                replace_indices = np.where(best_old > best)[0]
                # Copy in the old ones that were better than the current ones
                prob[replace_indices,:] = prob_old[replace_indices,:]

        # Overlaps updated for next pass, save them
        previous_scores = previous_updated

        if score_idx < N:
            score_idx += 1  # Start on next score block
        else:
            # No more new frames.  All done when we have finished processing all of the
            # previous ones
            more = len(previous_scores) > 0

        """Probabilities have now been merged across spectrograms for the section of the spectrogram
        that will not be overlapped in subsequent analysis spectrograms.  Determine what calls
        lie within this section.
        """
        # Process calls that were within the last section
        for cidx in range(catsN-1):
            scanners[cidx].next_block(prob[:,cidx])

        if debug >= 5:
            plot_detection_signals(prob, scanners, frameadv_s)

        True

    # Merge lists and sort (we could do this faster with a merge sort)
    results = []
    for s in scanners:
        results.extend(s.detections)
    results.sort(key=lambda d : d.start)

    return results

def plot_detection_signals(prob, scanners, frameadv_s=None):
    """
    plotit - debug function to show current signal and scanner states
    :param prob: probability matrix
    :param scanners: list of FindDetections objects that have been run on the current probability matrix
    """

    # determine range we are plotting
    start = scanners[0].block_idx * scanners[0].frames_n
    stop = start + scanners[0].frames_n
    # new figure with data and legend
    if frameadv_s:
        title = 'block %.3f - %.3f sec'%(start*frameadv_s, (stop-1)*frameadv_s)
    else:
        title = 'block %d-%d'%(start, stop-1)
    plt.figure(title)
    plt.plot(range(start, stop), prob[:,:-1])
    plt.ylim((0,1))  # Normalize range
    plt.legend([s.label for s in scanners])
    plt.xlabel('Frame index')
    plt.ylabel('P(class|observation)')
    # annotate what we've learned
    for s in scanners:
        didx = len(s.detections) - 1  # last one
        # iterate backwards until we run out or the end is no longer on the current plot
        while didx >= 0:
            if s.detections[didx].end < start:
                break  # ends before current block
            else:
                plt.plot(s.detections[didx].start, s.detections[didx].scores[0], '<')
                plt.plot(s.detections[didx].end, s.detections[didx].scores[-1], '>')
                didx -= 1

        # Show last low water mark (if on plot)
        if not s.first_low is None:
            plt.plot(s.first_low, s.scores[0], 'v')



class FindDetections:
    """
    FindDetections
    Class for processing detection signals that have been segmented into blocks
    Uses a low- and high- water mark detection algorithm.  When we pass the high
    water mark a detection is found and is extended backwards and forwards in time
    until the end of signal or until we cross the low water mark.
    """
    def __init__(self, label, low_thr, high_thr, frames_n):
        """

        :param label:  label that will be used for this class
        :param low_thr:
        :param high_thr:
        :param frames_n: Size of block in frames.
        """

        self.label = label
        self.low = low_thr
        self.high = high_thr
        self.frames_n = frames_n
        self.block_idx = -1

        self.first_low = None  # Set when we cross over the low water mark, reset when we fall below
        self.in_detection = False

        self.detections = []
        self.scores = []

    def get_detections(self):
        """
        get_detections - Return detection list
        :return:  List of TupDetection
        """
        return self.detections

    def next_block(self, p: object) -> object:
        """
        Given a block of probabilities, find detections or parts thereof
        :param p:
        """

        self.block_idx += 1
        offset = self.block_idx * self.frames_n
        idx = 0

        more = True
        while more:
            if self.first_low is None and p[idx] >= self.low:
                # Track possible start of low water region
                self.first_low = offset + idx
            if p[idx] < self.low:
                # We are below the low-water mark.
                if self.in_detection:
                    # Transitioning out of a detection as below low-water mark
                    d = TupDetection(self.label, self.first_low, offset + idx - 1, scores=self.scores)
                    self.detections.append(d)
                    self.in_detection = False
                # Regardless of whether or not there was a detection, reset all
                # the low-water active tracking variables as we are below the low-water mark
                self.first_low = None
                self.scores = []

            # If we are tracking a potential detection, retain its scores
            if not self.first_low is None:
                self.scores.append(p[idx])

            # If we cross the high water mark, we are in a detection
            if p[idx] >= self.high:
                self.in_detection = True

            # Move to next detection probability and determine if there's more to process
            idx += 1
            more = idx < np.min((self.frames_n, p.shape[0]))


def pred_to_audition(path, sep , engine ):
    '''
    path is for the predictions table pred_table
    sep should be equal to ";"
    engine should be equal to None
    '''
    
    pred_table = pd.read_csv(path, sep=sep, header=0, engine = engine) 
    pred_table.rename(columns={'Label':'Name'}, inplace=True)
    
    audition_table = pred_table[["Name", "Start"]]
    
    
    f = lambda x: pre.convert_secs_to_fulltime(x["Start"]) 
    audition_table["Start"] = pred_table.apply(f, axis=1)
    
    
    f = lambda x: pre.convert_secs_to_fulltime(x["Duration"])    
    audition_table["Duration"] = pred_table.apply(f, axis=1)
    audition_table["Time Format"] = "decimal"
    
    audition_table["Type"] = "Cue"
    audition_table["Description"] = ""
    
    return audition_table
    
     

