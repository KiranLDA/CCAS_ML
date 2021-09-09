'''
'''
import sys
sys.path.append("/home/mathieu/Documents/Git/CCAS_ML")

import numpy as np
import glob
import os
import tensorflow as tf
import pandas as pd
import pickle
import ntpath

from predation_params import *


class Evaluate:
    def __init__(self, label_list, 
                 prediction_list, 
                 model,
                 data, 
                 low_thresh,
                 high_thresh,
                 noise_label = "noise", 
                 IoU_threshold = 0.5,  
                 GT_proportion_cut = 0, 
                 ):
        '''
        label_list: list of the paths to the label files
        prediction_list: list of the paths to the prediction files
        model: name of the model used for the training
        run: name of the training
        noise_label: name of the noise/FP class
        IoU_threshold: sets the minimum proportion of overlap there should be between a labelled call and a prediction for them to be considered a match.
        low_thresh and high_thresh: thresholds used to create the prediction files, they're only used here to determine the file name.
        call_analysis: changes the mode of analysis:
            - "all_types_combined": all call_types except noise are treated as the same call type
            - "NAME_OF_CALL_TYPE": only calls of type NAME_OF_CALL_TYPE will be processed
            - "call_type_by_call_type": loops over all true call types; 
            for each step only calls of this type are processed, 
            and they are considered as accurate predictions if they are predicted as any true call.
            - "normal": normal analysis, with separation of focal and non-focal
        '''
        
        from predation_params import headers
        
        all_files = [label_list, prediction_list]
        self.headers = headers
        
        # Checking that all files have the same call types
        self.call_types = None
        for l in range(len(all_files)):
            if l == 1:
                (self.headers).add('scores')
            for file in all_files[l]:
                table = pd.read_csv(file, delimiter=';')
                table = table.loc[table["Label"] != "Label"] #eliminates the headers
                column_names = table.columns.tolist()
                call_types = set(column_names) - self.headers
                if self.call_types is None: # defines the recognised call types
                    self.call_types = call_types
                else: # checks that the call types used are the same in all files.
                    if self.call_types != call_types:
                        list_of_calls = [s for s in self.call_types]
                        err_calls = " ".join(list_of_calls)
                        raise ValueError("Call types inconsistent in " + file + ", expected " + err_calls)
                if(len(self.headers.intersection(set(column_names)).union({"scores"})) != len(self.headers.union({"scores"}))):
                    raise ValueError("File %s missing headers %s"%(file, self.headers - set(column_names))) # checks that all the headers are there. There have been issues with the 'scores' column being absent in some files, so scores is ignored here. 
        
        self.GT_path = sorted(label_list)
        self.pred_path = sorted(prediction_list)
        self.IoU_threshold = IoU_threshold
        if noise_label in self.call_types:
            self.noise_label = noise_label
        else:
            raise ValueError("Unknown noise label %s"%(noise_label))
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        
        self.model = model
        self.data = data
        self.min_call_length = {}
        if len(self.min_call_length.keys()) ==  0:
            for call in self.call_types:
                self.min_call_length[call] = 0
                
        # self.min_call_length = {"agg":7, "al":9, "cc":16, "ld":24, "mo":15, "sn":5, "soc":10}
        self.frame_rate = 200
        self.GT_proportion_cut = GT_proportion_cut
    

    
    
    def get_call_ranges(self, tablenames):
        ''' Like get_call_ranges, but separates focal and nonfocal calls in two
        different tables. The non-focal calls are marked as such in the label 
        tables. There is no distinction of focal and non-focal for prediction
        files.'''
        
        from predation_params import nonfoc_tags, start_labels, stop_labels, skipon_labels, skipoff_labels
     
        calls_indices = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))   # This table is defined for both ground truth and prediction. In both cases, rows represent recording files, columsn represent call types, and each cell is a list of tuples representing the beginning and end of each call of that type in that file  
        foc = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))          # This table represents wheter the calls defined above are focal or not. Since only the ground truth can be non-focal, only the GT version is used.
        for i in range(len(tablenames)):
            for call in call_types:
                calls_indices.at[i,call] = []
                foc.at[i,call] = []
        
        for i in range(len(tablenames)):
            skipped = 0
            table = pd.read_csv(tablenames[i], delimiter=';') 
            row = 0
            # go to Start
            while(row < len(table) and not table.Label[row] in start_labels): # All calls before start are skipped
                row += 1
                skipped += 1
                
            # main loop
            table_end = False # whether we've reached the 'End' label
            while(row < len(table) and not table_end):
                if table.Label[row] in skipon_labels: # parts of some files must be skipped. Those are surrounded by two rows with the labels 'skipon' and 'skipoff'.
                    while(table.Label[row] not in  skipoff_labels and row < len(table) and not table_end):
                        row += 1
                        skipped += 1
                else:
                    if table.Label[row] in stop_labels:
                        table_end = True
                        
                    # Determining the call type for that row
                    actual_call = None
                    to_be_skipped = False # There should be one 'True' per line. In any other case, the line will be skipped and counted as such.
                    for call in self.call_types:
                        if (table.at[row,call]):  
                            if(actual_call is None):
                                actual_call = call
                            else:
                                to_be_skipped = True
                    if(actual_call is None):
                        to_be_skipped = True
                    if(to_be_skipped):
                        skipped += 1
                    else:
                        # the beginning and end times for that call are added to the appropriate list 
                        # (or the list is created if this is the first call of that type in the file), 
                        # depending on whether a nonfocal tag is present in the label.
                        calls_indices.at[i,actual_call].append((table.Start[row], table.End[row]))
                        if any(word in table.Label[row] for word in nonfoc_tags):
                            foc.at[i,actual_call].append(False)
                        else:
                            foc.at[i,actual_call].append(True)                        
                    row += 1
            
            # Everything after the end is skipped
            while(row < len(table)):
                skipped += 1
                row += 1
            #print(str(skipped) + " out of " + str(len(table)) + " entries were skipped in "+ ntpath.basename(tablenames[i]))
            
        return [calls_indices,foc]
    


    def get_min_call_length(self, gt_indices):
        for call in self.call_types:
            if call in self.min_call_length.keys():
                duration = []
                for idx in range(len(gt_indices)):
                    for call_num in range(len(gt_indices.at[idx,call])):
                        duration.append(gt_indices.at[idx,call][call_num][1] - gt_indices.at[idx,call][call_num][0])
                duration.sort()
                shortest_call_allowed = int(len(duration) * self.GT_proportion_cut) 
                self.min_call_length[call] = duration[shortest_call_allowed]   
                



    def pairing(self, call_start, call_end, pred_start, pred_end, thr):
        '''Checks whether the call and the prediction can be considered as simultaneous,
        i.e. if the ratio of their intersection and their union is greater than
        the threshold thr'''
        paired = False
        if(call_start < pred_end and pred_start < call_end):
            intersection = min(call_end, pred_end) - max(call_start, pred_start)
            union = max(call_end, pred_end) - min(call_start, pred_start)
            if(union > 0 and intersection / union > thr):
                paired = True
        return paired

    
    
    def match_prediction_to_labels(self, gt_indices, foc, pred_indices):
        '''Generates a matrix that matches real calls with the detected ones.
        The first dimension is the type of call, the second dimension is the 
        type of detected call, and the third dimension is the number of the
        recording. The goal is to associate all predictions with the most relevant 
        GT call, before considering any question of whether the call is focal,
        if it's long enough, etc. Later, those calls or predictions that don't 
        fit a given selection criterion can be removed by pairs.
        If the mth call of type I for recording idx is detected as the nth call
        of type J, then match[I,J,idx] will contain at least the couple [m,n].
        If the call is not detected at all, then the last column for that recording 
        will have [NaN,m] at line I. Similarly, if the detection does not match
        a call, then the last lign for that recording will have [m,NaN] at column J.
        A similar entity loose_match is also returned, which corresponds to match
        with a pairing ratio of 0 (the call and the prediction can be match if 
        there is the least amount of overlap between them)
        In the current version, a labelled call can be matched to two different 
        predictions if two or more predictions were made for the time of the call,
        and none of them is of the right label. Similarly, a prediction can be 
        matched  to two or more simultaneous labelled calls if none of them are
        of the same type.'''
        # Creating a list whose dimensions are number of samples, number of call types, number of possible predictions,
        match = []
        loose_match = []
        while len(match) < np.size(gt_indices,0):
            match.append([])
            loose_match.append([])
        col = self.call_types.copy() # This avoids modifying self.call_types
        col.add('FN')
        row = self.call_types.copy()
        # A variable paired_pred is created to keep scores of which predictions have already been paired to a match,
        # so a labeled call cannot be paired with a simultaneous prediction if a better prediction has already been made.
        # This is why the matching starts with the true positives.
        paired_pred = pd.DataFrame(columns = self.call_types, index = range(pred_indices.shape[0]))
        print("Matching predictions ...")
        for idx in range(np.size(pred_indices,0)):
            for pred in self.call_types:
                if isinstance(pred_indices.at[idx,pred], list):
                    paired_pred.at[idx,pred] = np.zeros(len(pred_indices.at[idx,pred]), dtype = bool)                    
                else:
                    paired_pred.at[idx,pred] = []
        paired_call = pd.DataFrame(columns = self.call_types, index = range(gt_indices.shape[0]))
        for idx in range(np.size(gt_indices,0)):
            for call in self.call_types:
                if isinstance(gt_indices.at[idx,call], list):
                    paired_call.at[idx,call] = np.zeros(len(gt_indices.at[idx,call]), dtype = bool)
                else:
                    paired_call.at[idx,call] = []
        
        # Finding the true positives
        print("Finding true positives...")
        # for idx in [0,1]:
        for idx in range(np.size(gt_indices,0)):            
            #print(idx)
            match[idx] = pd.DataFrame(columns = col, index = row)
            loose_match[idx] = pd.DataFrame(columns = col, index = row)
            for c in col:
                match[idx][c] = [[] for r in row]
                loose_match[idx][c] = [[] for r in row]
            # We first loop over the prediction to determine which are true positives.
            # If they are, they are removed as potential matches for other labelled calls.
            for pred in self.call_types:
                if pred in self.min_call_length.keys():
                    min_length = self.min_call_length[pred]
                    min_length = 0
                else:
                    min_length = 0
                if isinstance(pred_indices.at[idx,pred], list):
                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                        if isinstance(gt_indices.at[idx,pred], list):
                            for call_nb in range(len(gt_indices.at[idx,pred])):
                                call_start = gt_indices.at[idx,pred][call_nb][0]
                                call_end = gt_indices.at[idx,pred][call_nb][1]
                                pred_start = pred_indices.at[idx,pred][pred_nb][0]
                                pred_end = pred_indices.at[idx,pred][pred_nb][1]
                                paired = self.pairing(call_start, call_end, pred_start, pred_end, 0)
                                if paired: # The first pairing is a loose match
                                    loose_match[idx].at[pred,pred].append((call_nb,pred_nb))
                                    better_paired = self.pairing(call_start, call_end, pred_start, pred_end, self.IoU_threshold)
                                    if better_paired and foc.at[idx,pred][call_nb] and pred_end - pred_start > min_length: # Pairing for the strict match.
                                        # if pred_end - pred_start > min_length: #/ (self.frame_rate - 1): # checking that the length of the prediction is reasonable
                                        paired_pred.at[idx,pred][pred_nb] = True
                                        paired_call.at[idx,pred][call_nb] = True   
                                        match[idx].at[pred,pred].append((call_nb,pred_nb))
        
        # Finding the wrong detections and false negatives:
        print("Finding false negatives:")
        # for idx in [0,1]:
        for idx in range(np.size(gt_indices,0)):            
            #print(idx)
            for call in self.call_types:
                if isinstance(gt_indices.at[idx,call], list):
                    for call_nb in range(len(gt_indices.at[idx,call])):
                        if paired_call.at[idx,call][call_nb]: # Only the labelled calls that haven't been matched yet are considered at this stage.
                            matched_call = True
                        else:
                            matched_call = False
                            for pred in self.call_types:
                                if pred in self.min_call_length.keys():
                                    min_length = self.min_call_length[pred]
                                else:
                                    min_length = 0                                
                                if (isinstance(pred_indices.at[idx,pred], list) and pred != call):
                                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                                        if not paired_pred.at[idx,pred][pred_nb]:
                                            call_start = gt_indices.at[idx,call][call_nb][0]
                                            call_end = gt_indices.at[idx,call][call_nb][1]
                                            pred_start = pred_indices.at[idx,pred][pred_nb][0]
                                            pred_end = pred_indices.at[idx,pred][pred_nb][1]
                                            paired = self.pairing(call_start, call_end, pred_start, pred_end, 0)
                                            if paired:
                                                loose_match[idx].at[call,pred].append((call_nb,pred_nb))  
                                                better_paired = self.pairing(call_start, call_end, pred_start, pred_end, self.IoU_threshold)
                                                if better_paired and foc.at[idx,call][call_nb] and pred_end - pred_start > min_length:
                                                    paired_pred.at[idx,pred][pred_nb] = True
                                                    matched_call = True
                                                    match[idx].at[call,pred].append((call_nb,pred_nb))                                                
                                                
                        if not matched_call:
                            loose_match[idx].at[call,'FN'].append((call_nb,np.nan)) 
                            call_start = gt_indices.at[idx,call][call_nb][0]
                            call_end = gt_indices.at[idx,call][call_nb][1]                            
                            if foc.at[idx,call][call_nb]:
                                match[idx].at[call,'FN'].append((call_nb,np.nan))
                                                
        # At this point all labelled calls have been matched or classified as false negatives.
        # Only the false positives still need to be marked.        
        print("Marking false positives")
        # for idx in [0,1]:
        for idx in range(np.size(pred_indices,0)):
            for pred in self.call_types:
                if isinstance(pred_indices.at[idx,pred], list):
                    if pred in self.min_call_length.keys():
                        min_length = self.min_call_length[pred]
                    else:
                        min_length = 0
                    for pred_nb in range(len(pred_indices.at[idx,pred])):                        
                        if not paired_pred.at[idx,pred][pred_nb]:     
                            loose_match[idx].at[self.noise_label,pred].append((np.nan,pred_nb)) 
                            pred_start = pred_indices.at[idx,pred][pred_nb][0]
                            pred_end = pred_indices.at[idx,pred][pred_nb][1]                            
                            if pred_end - pred_start > min_length:
                                match[idx].at[self.noise_label,pred].append((np.nan,pred_nb))
                            # FP are sorted in noise. One consequence of that is that the call type noise 
                            # can be properly identified and still increase the number of false positives.
                
        print("Finished FP marking!") 
        
        return match, loose_match 
    

    
    def precision(self, TPos,FPos):
        '''Precision is TP / (TP+FP)'''
        Prec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            if TPos[call] == 0 and FPos[call] == 0:
                Prec[call] = 1.0 # If the call was never predicted, the precision is set to 1.0 by convention.
            else:
                Prec[call] = TPos[call] / (TPos[call] + FPos[call])
        return Prec
    
    def lenient_precision(self, cm):
        ''' Like precision, but a match is considered a true positive as long 
        as the call is matched to an actual call type. This metric allows us to
        determine whether animal sounds are properly classified as such.'''
        Prec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            all_preds = sum(cm[call])
            if all_preds == 0:
                Prec2[call] = 1.0 # If the call was never predicted, the precision is set to 1.0 by convention.
            else:
                Prec2[call] = (all_preds - cm.at[self.noise_label,call]) / all_preds
        return Prec2
    
    def recall(self, TPos,FNeg):
        '''Recall is TP / (TP+FN)'''
        Rec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            if TPos[call] == 0 and FNeg[call] == 0:
                Rec[call] = 1.0 # If the call was never predicted, the precision is set to 1.0 by convention.
            else:
                Rec[call] = TPos[call] / (TPos[call] + FNeg[call])
        return Rec
    
    def lenient_recall(self, cm):
        ''' Like recall, but a match is considered a true positive as long 
        as the call is matched to an actual call type. This metric allows us to
        determine whether animal sounds are properly classified as such.'''        
        Rec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            GT = sum(cm.loc[call])
            if GT == 0:
                Rec2[call] = 1.0 # If the call is not present in the labelled data, the precision is set to 1.0 by convention.
            else:
                Rec2[call] = (GT - cm.at[call, 'FN']) / GT
        return Rec2
    
    
    def get_confusion_matrix(self, match):
        '''Generates the confusion matrix corresponding to the list of calls gt_indices 
        and the list of predicted calls pred_indices.
        If there are N types of calls, the confusion matrix's size is (N+1)*(N+1);
        the additional line and column corresponding to the false negatives and false 
        positives respectively, in the sense that the call wasn't detected at all 
        (and not merely detected as a different type of call), or that the detection 
        doesn't match an actual call.'''

        col = self.call_types.copy()
        col.add('FN')
        row = self.call_types.copy()
        cm = pd.DataFrame(columns = col, index = row)
        for call in row:
            for pred in col:
                cm.at[call,pred] = 0
        # The first loop compares the list of calls to all detected calls for every recording.
        for idx in range(len(match)):
            for call in  self.call_types:
                for pred in  self.call_types:
                    if isinstance(match[idx].at[call,pred], list):
                        cm.at[call,pred]+=len(match[idx].at[call,pred])
        # The second loop deals with the FN and FP.
        for idx in range(len(match)):
            for call in self.call_types:
                if isinstance(match[idx].at[call,'FN'], list):
                    cm.at[call,'FN'] += len(match[idx].at[call,'FN'])
        return cm
    
    
    def time_difference(self, match, gt_indices, pred_indices):
        '''Compute for every paired call and prediction the offset between their starting times'''
        offset = [None] *len(match)
        for idx in range(len(offset)):
            offset[idx] = pd.DataFrame(columns = self.call_types, index = self.call_types)
            for call in self.call_types:
                for pred in self.call_types:
                    offset[idx].at[call,pred] = []
                    for pair_nb in range(len(match[idx].at[call,pred])):
                        call_start = gt_indices[call][idx][match[idx].at[call,pred][pair_nb][0]][0]
                        detected_start = pred_indices[pred][idx][match[idx].at[call,pred][pair_nb][1]][0]
                        offset[idx].at[call,pred].append(call_start - detected_start)
        return offset
    
    
    def category_fragmentation(self, match, gt_indices, pred_indices):
        '''For every call, how many call types is it detected as?'''
        cat_frag = pd.DataFrame(columns = match.columns, index = match.index)
        # cat_frag = pd.DataFrame(columns = self.call_types, index=range(len(self.GT_path)))#label_list)))
        for idx in range(len(match)):
            for call in self.call_types:
                cat_frag.at[idx,call] = match.at[idx,call].copy()
                if call != "noise":
                    for call_num in range(len(match.at[idx,call])):
                        cat_frag.at[idx,call][call_num] = 0
                        if match.at[idx,call][call_num] == match.at[idx,call][call_num]:
                            call_types_list = set()
                            for match_num in range(len(match.at[idx,call][call_num])):
                                next_type = match.at[idx,call][call_num][match_num][0]
                                if next_type not in call_types_list:
                                    call_types_list.add(next_type)
                                    cat_frag.at[idx,call][call_num] += 1
        return cat_frag
                           
                        
    def time_fragmentation(self, match, gt_indices, pred_indices):
        '''How many fragments the call is detected as, indepedently of categories.
        All segments of the call for which one or more prediction have been found 
        is taken into account'''
        time_frag = pd.DataFrame(columns = match.columns, index = match.index)
        for idx in range(len(match)):
            for call in self.call_types:
                time_frag.at[idx,call] = match.at[idx,call].copy()
                if call != "noise":
                    for call_num in range(len(match.at[idx,call])):
                        time_frag.at[idx,call][call_num] = 0
                        if match.at[idx,call][call_num] == match.at[idx,call][call_num]:
                            if match.at[idx,call][call_num][0][0] == 'FN':
                                time_frag.at[idx,call][call_num] = 0
                            else:
                                matched_pred = []
                                for match_num in range(len(match.at[idx,call][call_num])):
                                    matched_pred.append(pred_indices.at[idx,match.at[idx,call][call_num][match_num][0]][match.at[idx,call][call_num][match_num][1]])
                                matched_pred.sort()
                                t = 0 # end time of the current predicted call
                                f = 0 # number of fragments
                                p = 0 # pair number
                                tmax = max(matched_pred, key = lambda i : i[0])[1] # the latest end of a predicted call that was matched to the GT call
                                while(t < tmax and p < len(matched_pred)):
                                    t = matched_pred[p][1]
                                    p += 1
                                    while(p < len(matched_pred) and matched_pred[p][0] < t):
                                        t = matched_pred[p][1]
                                        p += 1
                                    f += 1
                                time_frag.at[idx,call][call_num] = f
        return time_frag
    
    def output(self, gt_indices, foc, pred_indices, match, call_match, pred_match, cm, prec, lenient_prec, rec, lenient_rec, offset, cat_frag, time_frag):
        '''
        Takes all the metrics previously computed and create a summary of the most important information.
        This includes:
        - number of files with number of focals
        - call dictionary
        - recording with the worst results
        - most confused call types
        - greatest difference between prec and lenient_prec
        - greatest difference between rec and lenient_rec
        - greatest offset
        - greatest negative offset
        - most ambiguous call
        - most fragmented call
        '''
        files = list(gt_indices.index)
        file_dict = dict()
        self.call_types = set(gt_indices.columns)
        for idx in range(len(gt_indices)):
            file_dict[files[idx]] = 0
            for call in self.call_types:
                file_dict[files[idx]] += sum(foc.iloc[idx][call])
                
        call_dict = dict()
        for call in self.call_types:
            call_dict[call] = 0
            for file in files:
                call_dict[call] += len(gt_indices.at[file, call])
                
        
        

    

    def main(self):
           
        print( "Formatting ground truths...")
        gt_indices, foc = self.get_call_ranges(self.GT_path)
        if isinstance(self.GT_proportion_cut, float):
            self.get_min_call_length(gt_indices)
        print( "Formatting predictions...")
        pred_indices, _ = self.get_call_ranges(self.pred_path) 
        # Matches the labelled data with the predicted calls. loose_match is equivalent to a match with an intersection of union threshold of 0.
        match, loose_match = self.match_prediction_to_labels(gt_indices, foc, pred_indices)
        # loose_match pairs together ground truth and prediction regardless of the quality of either. We then remove the pairs that shouldn't be processed in the analysis (too short calls, non-focal, unproperly matched pairs, etc.)
        # match = self.data_selection(gt_indices, foc, match, loose_match)
        
        call_match = pd.DataFrame(columns = self.call_types, index=range(len(gt_indices)))
        pred_match = pd.DataFrame(columns = self.call_types, index=range(len(gt_indices)))
        # for idx in [0,1]:
        for idx in range(np.size(gt_indices,0)): 
            for call in match[idx].index:
                if call != self.noise_label:
                    call_match.at[idx,call] = [np.NaN] * len(gt_indices.at[idx,call])
                    for pred in match[idx].columns:
                        for match_nb in range(len(match[idx].at[call,pred])):
                            if call_match.at[idx,call][match[idx].at[call,pred][match_nb][0]] != call_match.at[idx,call][match[idx].at[call,pred][match_nb][0]]:
                                call_match.at[idx,call][match[idx].at[call,pred][match_nb][0]] = [(pred, match[idx].at[call,pred][match_nb][1])]
                            else:
                                call_match.at[idx,call][match[idx].at[call,pred][match_nb][0]].append((pred, match[idx].at[call,pred][match_nb][1]))
            # nan means this is an ignored GT (typically, non-focal)
                        
            for pred in match[idx].columns:
                if pred != "FN":
                    pred_match.at[idx,pred] = [np.NaN] * len(pred_indices.at[idx,pred])
                    for call in match[idx].index:
                        for match_nb in range(len(match[idx].at[call,pred])):
                            if pred_match.at[idx,pred][match[idx].at[call,pred][match_nb][1]] != pred_match.at[idx,pred][match[idx].at[call,pred][match_nb][1]]:
                                pred_match.at[idx,pred][match[idx].at[call,pred][match_nb][1]] = [(pred, match[idx].at[call,pred][match_nb][0])]
                            else:
                                pred_match.at[idx,pred][match[idx].at[call,pred][match_nb][1]].append((pred, match[idx].at[call,pred][match_nb][0]))
        
        cm = self.get_confusion_matrix(match)
        self.call_types.remove(self.noise_label) # As the noise is also the label for the false positives, it would cause problems to leave it in the data at this point.
        TPos = dict.fromkeys(self.call_types,0)
        FPos = dict.fromkeys(self.call_types,0) 
        FNeg = dict.fromkeys(self.call_types,0) 
        for call in self.call_types:
            TPos[call] = 0
            FPos[call] = 0
            FNeg[call] = 0
            for pred in self.call_types:
                if call!= pred:
                    FPos[call] += cm.at[pred,call]
                    FNeg[call] += cm.at[call,pred]
                else:
                    TPos[call] += cm.at[call,pred]
            FPos[call] += cm.at[self.noise_label,call]
            FNeg[call] += cm.at[call,"FN"]
        Prec = self.precision(TPos,FPos)
        Rec = self.recall(TPos,FNeg)
        lenient_Rec = self.lenient_recall(cm)
        lenient_Prec = self.lenient_precision(cm)
        offset = self.time_difference(match, gt_indices, pred_indices)
        cat_frag = self.category_fragmentation(call_match, gt_indices, pred_indices)
        time_frag = self.time_fragmentation(call_match, gt_indices, pred_indices)   
        
        # preparing the metrics for saving
        for i in range(len(gt_indices)):
            time_frag.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            cat_frag.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            gt_indices.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            pred_indices.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            foc.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            call_match.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            pred_match.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            cat_frag.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True)
            time_frag.rename(index={i: os.path.basename(self.pred_path[i])}, inplace=True) 
        
        # # store the outputs    
        # output = dict()
        # output["Label_Indices"] = gt_indices
        # output["Prediction_Indices"] = pred_indices
        # output["Focal"] = foc
        # output["Matching_Table"] = match
        # output["Matching_Table-Labels_Sorted"] = call_match
        # output["Matching_Table-Predictions_Sorted"] = pred_match
        # output["Confusion_Matrix"] = cm
        # output["Precision"] = Prec
        # output["Lenient_Precision"] = lenient_Prec
        # output["Recall"] = Rec
        # output["Lenient_Recall"] = lenient_Rec
        # output["Time_Difference"] = offset
        # output["Category_Fragmentation"] = cat_frag
        # output["Time_Fragmentation"] = time_frag        

        # Creating the metrics folder
        main_dir =  os.path.join("/media/mathieu/Elements/code/KiranLDA/results/", self.model, self.data, "metrics")
        metrics_folder = os.path.join(main_dir, str(self.GT_proportion_cut), str(self.low_thresh), str(self.high_thresh))
        directories = [main_dir,
                        os.path.join(main_dir, str(self.GT_proportion_cut)),
                        os.path.join(main_dir, str(self.GT_proportion_cut), str(self.low_thresh)),
                        metrics_folder]
        for diri in directories:
            if not os.path.exists(diri):
                os.mkdir(diri)
        
        
        # Saving the metrics
        output_files = self.output(gt_indices, foc, pred_indices, match, call_match, pred_match, cm, prec, lenient_prec, rec, lenient_rec, offset, cat_frag, pred_indices)
        with open(os.path.join(metrics_folder, 'Label_Indices.p'), 'wb') as fp:
            pickle.dump(gt_indices, fp)
        with open(os.path.join(metrics_folder, 'Label_Indices.p'), 'wb') as fp:
            pickle.dump(gt_indices, fp)
        with open(os.path.join(metrics_folder, 'Prediction_Indices.p'), 'wb') as fp:
            pickle.dump(pred_indices, fp)
        with open(os.path.join(metrics_folder, 'Focal.p'), 'wb') as fp:
            pickle.dump(foc, fp)        
        with open(os.path.join(metrics_folder, 'Matching_Table.p'), 'wb') as fp:
            pickle.dump(match, fp)
        with open(os.path.join(metrics_folder, 'Matching_Table-Labels_Sorted.p'), 'wb') as fp:
            pickle.dump(call_match, fp)
        with open(os.path.join(metrics_folder, 'Matching_Table-Predictions_Sorted.p'), 'wb') as fp:
            pickle.dump(pred_match, fp)        
        cm.to_csv(os.path.join(metrics_folder, 'Confusion_matrix.csv'))
        with open(os.path.join(metrics_folder, 'Precision.p'), 'wb') as fp:
            pickle.dump(Prec, fp)
        with open(os.path.join(metrics_folder, 'Lenient_Precision.p'), 'wb') as fp:
            pickle.dump(lenient_Prec, fp)
        with open(os.path.join(metrics_folder, 'Recall.p'), 'wb') as fp:
            pickle.dump(Rec, fp)
        with open(os.path.join(metrics_folder, 'Lenient_Recall.p'), 'wb') as fp:
            pickle.dump(lenient_Rec, fp)            
        with open(os.path.join(metrics_folder, 'Time_Difference.p'), 'wb') as fp:
            pickle.dump(offset, fp)  
        with open(os.path.join(metrics_folder, 'Category_fragmentation.p'), 'wb') as fp:
            pickle.dump(cat_frag, fp)
        with open(os.path.join(metrics_folder, 'Time_Fragmentation.p'), 'wb') as fp:
            pickle.dump(time_frag, fp)


        # return output


def list_files(directory, ext=".txt"):
    "list_files(directory) - Grab all .txt or specified extension files in specified directory"
    files = glob.glob(os.path.join(directory, "*" + ext))
    files.sort()
    return files


if __name__=="__main__":
    # model = "NoiseAugmented_ProportionallyWeighted_NoOther_2020-10-14_03:12:32.817594"
    # model = "EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/"
    from predation_params import run_name, test_path, data, low_thresholds, high_thresholds, short_GT_removed
    # run = "new_run"
    # main_dir =  os.path.join("/media/mathieu/Elements/code/KiranLDA/results", model, run)
    main_dir =  test_path
    label_dir = os.path.join(main_dir, "label_table")
    label_list = list_files(label_dir)
    
    low_thresholds = [0.1]
    ''' The model makes a prediction for any time point in the wav file to be 
    associated with a given type call. When a time point reaches a prediction 
    score higher than an upper threshold (high_thresh), every point in the wav file 
    before and after it is considered as belonging to a predicted call, until
    a point for which the score is below a lower threshold (low_thresh). Thus 
    the closer the two thresholds are, the more calls will be predicted; the 
    lower the upper threshold, the more false positives; the higher the upper 
    threshold, the more false negatives. Such predictions have been run for 
    various combinations of lower and higher thresholds. This program computes
    various evaluation metrics for each set of predictions.
    '''
    for GT_proportion_cut in short_GT_removed: #,0.005,0.01,0.015,0.02,0.025,0.03]:
        for low_thresh in low_thresholds:
            for high_thresh in high_thresholds:
                if low_thresh < high_thresh:
                    print([GT_proportion_cut, low_thresh, high_thresh])
                    
                    '''There are eleven classes of meerkat calls, these seven are 
                    the calls we're interested in.There are also two classes that
                    corresponds to artefacts of the recording method ('beep' and 
                    'synch'), a noise class, and an other class ('oth') for the 
                    calls that couldn't be properly labelled'''
                    
                    #true_calls = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc'}
                    
                    results_dir = os.path.join(main_dir, "results_per_threshold", str(low_thresh), str(high_thresh))
                    prediction_list = list_files(results_dir, str(low_thresh) + "-" + str(high_thresh) + ".txt") #list of the paths where the prediction are stored, each file corresponds to one recording session for one animal.
                    
                    if len(prediction_list) != len(label_list):
                        raise ValueError("Numbers of files don't match")
                    # for CALL in true_calls:
                        # evaluate = Evaluate(label_list, prediction_list, noise_label = "noise", IoU_threshold = 0.5, gap_threshold = 5, high_thresh = thresh, call_analysis = "oth")
                    evaluate = Evaluate(label_list, prediction_list, model = run_name, data = data, low_thresh = low_thresh, high_thresh = high_thresh,  GT_proportion_cut = GT_proportion_cut)
                    #model = model, run = run, low_thresh = low_thresh, high_thresh = thresh,
                    '''call_analysis can be: 
                        - "all_types_combined": all call_types except noise are treated as the same call type
                        - "NAME_OF_CALL_TYPE": only calls of type NAME_OF_CALL_TYPE will be processed
                        - "call_type_by_call_type": loops over all true call types; 
                        for each step only calls of this type are processed, 
                        and they are considered as accurate predictions if they are predicted as any true call.
                        - "normal": normal analysis, with separation of focal and non-focal
                        By default, the analysis is set to normal.
                    '''
                    evaluate.main()
