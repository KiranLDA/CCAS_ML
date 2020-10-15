#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:27:18 2020
@author: Mathieu
"""
import numpy as np
import glob
import os
import tensorflow as tf
import pandas as pd
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.Session(config=config)


class Evaluate:
    def __init__(self, label_list, prediction_list, noise_label = "noise", IoU_threshold = 0.5, gap_threshold = 5, call_analysis = "normal"):
        all_files = [label_list, prediction_list]
        self.headers = set(['Label', 'Duration', 'Start', 'End', 'scores'])
        # Checking that all files have the same call types
        self.call_types = None
        for l in all_files:
            for file in l:
                z = pd.read_csv(file, delimiter=';')
                column_names = z.columns.tolist()
                call_types = set(column_names) - self.headers
                if self.call_types is None:
                    self.call_types = call_types
                else:
                    if self.call_types != call_types:
                        list_of_calls = [s for s in self.call_types]
                        err_calls = " ".join(list_of_calls)
                        raise ValueError("Call types inconsistent in " + file + ", expected " + err_calls)
                if(len(self.headers.intersection(set(column_names + ['scores']))) != len(self.headers)):
                    raise ValueError("File %s missing headers %s"%(file, self.headers - set(column_names)))
        
        self.GT_path = sorted(label_list)
        self.pred_path = sorted(prediction_list)
        self.IoU_threshold = IoU_threshold
        self.gap_threshold = gap_threshold
        if noise_label in self.call_types:
            self.noise_label = noise_label
        else:
            raise ValueError("Unknown noise label %s"%(noise_label))
        self.noise_label = "noise" #"noncall"
        self.no_call = set(["noise", "beep", "synch"])
        self.true_call= set(self.call_types.difference(self.no_call))
        self.call_analysis = call_analysis
        
    def get_call_ranges(self, tablenames):
        skipped = 0
        x = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))
        calls = list(self.call_types)
        calls.sort()
        for i in range(len(tablenames)):
            skipped = 0
            extract = 0
            table = pd.read_csv(tablenames[i], delimiter=';') 
            # table['Start'] = table['Start'].astype('float')
            # table['Duration'] = table['Duration'].astype('float')
            # table['End'] = table['End'].astype('float')
            # for call in calls:
            #     table[call] = table[call].astype('bool')
            # df['column_name'] = df['column_name'].astype('bool')
            row = 0
            # go to Start
            while(row < len(table) and not table.Label[row] in ['START','start']):
                row += 1
                skipped += 1
            # main loop
            table_end = False # whether we've reached the 'End' label
            while(row < len(table) and not table_end):
                if table.Label[row] == 'END':
                    table_end = True
                cidx0 = None
                to_be_skipped = False # There should be one 'True' per line. In any other case, the line will be skipped and counted as such.
                for cidx in range(len(self.call_types)):
                    if (table[calls[cidx]][row]):  #  == True or table[calls[cidx]][row] == 'True'
                        if(cidx0 is None):
                            cidx0 = cidx
                        else:
                            to_be_skipped = True
                if(cidx0 is None):
                    to_be_skipped = True
                if(to_be_skipped):
                    skipped += 1
                else:
                    if(x[calls[cidx0]][i] != x[calls[cidx0]][i]):
                        x[calls[cidx0]][i] = [(table.Start[row], table.End[row])]
                    else:
                        x[calls[cidx0]][i].append((table.Start[row], table.End[row]))
                    extract +=1
                row += 1
            
            # Everything after the end is skipped
            while(row < len(table)):
                skipped += 1
                row += 1
            
            # print("File %d=%s skipped %d extract %d if %d else %d"%(
            #     i, tablenames[i], skipped, extract, ifnb, elsenb))
        return x
    
    
    def get_nonfoc_and_foc_calls(self, tablenames):
        nonfoc_tags = ["NONFOC", "nf"]    
        skipped = 0
        
        foc = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))
        nonfoc = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))
        calls = list(self.call_types)
        calls.sort()
        for i in range(len(tablenames)):
            skipped = 0
            extract = 0
            table = pd.read_csv(tablenames[i], delimiter=';') 
            row = 0
            # go to Start
            while(row < len(table) and not table.Label[row] in ['START','start']):
                row += 1
                skipped += 1
            # main loop
            table_end = False # whether we've reached the 'End' label
            while(row < len(table) and not table_end):
                if table.Label[row] == 'skipon':
                    while(table.Label[row] != 'skipoff' and row < len(table) and not table_end):
                        row += 1
                        skipped += 1
                else:
                    if table.Label[row] == 'END':
                        table_end = True
                    cidx0 = None
                    to_be_skipped = False # There should be one 'True' per line. In any other case, the line will be skipped and counted as such.
                    for cidx in range(len(self.call_types)):
                        if (table[calls[cidx]][row]):
                            if(cidx0 is None):
                                cidx0 = cidx
                            else:
                                to_be_skipped = True
                    if(cidx0 is None):
                        to_be_skipped = True
                    if(to_be_skipped):
                        skipped += 1
                    else:
                        if any(word in table.Label[row] for word in nonfoc_tags):
                            if(nonfoc[calls[cidx0]][i] != nonfoc[calls[cidx0]][i]):
                                nonfoc[calls[cidx0]][i] = [(table.Start[row], table.End[row])]
                            else:
                                nonfoc[calls[cidx0]][i].append((table.Start[row], table.End[row]))
                            extract +=1
                        else:
                            if(foc[calls[cidx0]][i] != foc[calls[cidx0]][i]):
                                foc[calls[cidx0]][i] = [(table.Start[row], table.End[row])]
                            else:
                                foc[calls[cidx0]][i].append((table.Start[row], table.End[row]))
                            extract +=1                        
                    row += 1
            
            # Everything after the end is skipped
            while(row < len(table)):
                skipped += 1
                row += 1
            
            nonfocfoc = [nonfoc,foc]
            
            if(self.call_analysis == "all_types_combined"):
                for x in range(len(nonfocfoc)):
                    y = pd.DataFrame(columns = ["call", "noncall"], index = range(len(nonfocfoc[x])))
                    for call in self.call_types:
                        if isinstance(nonfocfoc[x].at[i,call], list):
                            if call in self.true_call:
                                if isinstance(y.at[i,"call"], list):
                                    y.at[i,"call"]  = y.at[i,"call"] + nonfocfoc[x].at[i,call]
                                else:
                                    y.at[i,"call"] = nonfocfoc[x].at[i,call]
                            else:
                                if isinstance(y.at[i,"noncall"], list):
                                    y.at[i,"noncall"]  = y.at[i,"noncall"] + nonfocfoc[x].at[i,call]
                                else:
                                    y.at[i,"noncall"] = nonfocfoc[x].at[i,call]
                    nonfocfoc[x] = y
        return nonfocfoc
                
    def precision(self, TPos,FPos):
        '''Precision is TP / (TP+FP)'''
        Prec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Prec[call] = TPos.at[0,call] / (TPos.at[0,call] + FPos.at[0,call])
            if(np.isnan(Prec[call])):
                Prec[call] = 1
        Prec = pd.DataFrame(Prec, index=[0])
        return Prec

    def new_precision(self, cf):
        Prec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            all_preds = sum(cf[call])
            if all_preds == 0:
                Prec2[call] = np.nan
            else:
                Prec2[call] = (all_preds - cf.at[self.noise_label,call]) / all_preds
            if(np.isnan(Prec2[call])):
                Prec2[call] = 1
        Prec2 = pd.DataFrame(Prec2, index=[0])
        return Prec2
    
    def recall(self, TPos,FNeg):
        '''Recall is TP / (TP+FN)'''
        Rec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Rec[call] = TPos.at[0,call] / (TPos.at[0,call] + FNeg.at[call])
            if(np.isnan(Rec[call])):
                Rec[call] = 1
        Rec = pd.DataFrame(Rec, index=[0])
        return Rec
    
    def call_by_call_recall(self, cf):
        Rec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            GT = sum(cf.loc[call])
            if GT == 0:
                Rec2[call] = np.nan
            else:
                Rec2[call] = (GT - cf.at[call, 'FN']) / GT
            if(np.isnan(Rec2[call])):
                Rec2[call] = 1
        Rec2 = pd.DataFrame(Rec2, index=[0])
        return Rec2
    
    def pairing(self, call_start, call_end, pred_start, pred_end):
        '''Checks whether the call and the prediction are simultaneous'''
        paired = False
        if(call_start < pred_end and pred_start < call_end):
            intersection = min(call_end, pred_end) - max(call_start, pred_start)
            union = max(call_end, pred_end) - min(call_start, pred_start)
            if(union > 0 and intersection / union > self.IoU_threshold):
                paired = True
        return paired
    
    def match_prediction_to_labels(self, gt_indices, pred_indices):
        '''Generates a matrix that matches real calls with the detected ones.
        The first dimension is the type of call, the second dimension is the 
        type of detected call, and the third dimension is the number of the
        recording.
        If the mth call of type I for recording idx is detected as the nth call
        of type J, then match[J,I,idx] will contain at least the couple [n,m].
        If the call is not detected at all, then the last column for that recording 
        will have [NaN,m] at line I. Similarly, if the detection does not match
        a call, then the last lign for that recording will have [m,NaN] at column J.
        '''
        # Creating a list whose dimensions are number of samples, number of call types, number of possible predictions,
        match = []
        while len(match) < np.size(gt_indices,0):
            match.append([])
        col = self.call_types.copy()
        col.add('FN')
        row = self.call_types.copy()
        paired_pred = pd.DataFrame(columns = self.call_types, index = range(pred_indices.shape[0]))
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
        for idx in range(np.size(gt_indices,0)):
            print("******************* Finding True Positives *******************")
            print(idx)
            match[idx] = pd.DataFrame(columns = col, index = row)
            for c in col:
                match[idx][c] = [[] for r in row]
            for pred in self.call_types:
                if isinstance(pred_indices.at[idx,pred], list):
                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                        if isinstance(gt_indices.at[idx,pred], list):
                            for call_nb in range(len(gt_indices.at[idx,pred])):
                                call_start = gt_indices.at[idx,pred][call_nb][0]
                                call_end = gt_indices.at[idx,pred][call_nb][1]
                                pred_start = pred_indices.at[idx,pred][pred_nb][0]
                                pred_end = pred_indices.at[idx,pred][pred_nb][1]
                                paired = self.pairing(call_start, call_end, pred_start, pred_end)
                                if paired:
                                    paired_pred.at[idx,pred][pred_nb] = True
                                    paired_call.at[idx,pred][call_nb] = True
                                    match[idx].at[pred,pred].append([call_nb,pred_nb])
        
        # Finding the wrong detections and false negatives:
        for idx in range(np.size(gt_indices,0)):
            print("******************* Finding False Negatives *******************")
            print(idx)
            for call in self.call_types:
                if isinstance(gt_indices.at[idx,call], list):
                    for call_nb in range(len(gt_indices.at[idx,call])):
                        if paired_call.at[idx,call][call_nb]:
                            matched_call = True
                        else:
                            matched_call = False
                            for pred in self.call_types:
                                if (isinstance(pred_indices.at[idx,pred], list) and pred != call):
                                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                                        if not paired_pred.at[idx,pred][pred_nb]:
                                            # print([idx,call,call_nb,pred,pred_nb])
                                            call_start = gt_indices.at[idx,call][call_nb][0]
                                            call_end = gt_indices.at[idx,call][call_nb][1]
                                            pred_start = pred_indices.at[idx,pred][pred_nb][0]
                                            pred_end = pred_indices.at[idx,pred][pred_nb][1]
                                            paired = self.pairing(call_start, call_end, pred_start, pred_end)
                                            if paired:
                                                paired_pred.at[idx,pred][pred_nb] = True
                                                matched_call = True
                                                match[idx].at[call,pred].append([call_nb,pred_nb])                                            
                        if not matched_call:
                            match[idx].at[call,'FN'].append([call_nb,np.nan])
                                                
        # Finding the false positives:        
        for idx in range(np.size(pred_indices,0)):
            for pred in self.call_types:
                if isinstance(pred_indices.at[idx,pred], list):
                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                        if not paired_pred.at[idx,pred][pred_nb]:
                            match[idx].at[self.noise_label,pred].append([np.nan,pred_nb])  
                            # FP are sorted in noise. One consequence of that is that noise can be misidentified as noise.
                            
        return match

    def match_specific_call(self, gt_indices, pred_indices, match):
        match2 = []
        col = self.call_types.copy()
        col.add("Lab")
        while len(match2) < np.size(gt_indices,0):
            match2.append([])
        for idx in range(np.size(gt_indices,0)):
            call_list = []
            if gt_indices.at[idx,self.call_analysis] == gt_indices.at[idx,self.call_analysis]:
                table = pd.read_csv(self.GT_path[idx], delimiter=';')
                row = 0
                next_start = gt_indices.at[idx,self.call_analysis][0][0]
                '''we need to deal with start, stop , skipon and skipoff'''
                while next_start <= gt_indices.at[idx,self.call_analysis][len(gt_indices.at[idx,self.call_analysis])-1][0]:
                    while table.at[row,'Start'] < next_start:
                        row += 1
                    call_list.append(table.at[row,'Label'])
                    if len(call_list) < len(gt_indices.at[idx,self.call_analysis]):
                        next_start = gt_indices.at[idx,self.call_analysis][len(call_list)][0]
                    else:
                        next_start = table.at[len(table)-1,'End']
            match2[idx] = pd.DataFrame(columns = col, index = range(len(call_list)))
            if len(call_list) > 0:
                match2[idx] = match2[idx].assign(Lab = call_list)
                for call in self.call_types:
                    for call_num in range(len(match[idx].at[self.call_analysis,call])):
                        if call in call_list[call_num]:
                            match2[idx].at[call_num, call] = 'TP'
                        else:
                            match2[idx].at[call_num, call] = 'FP'
                    for row in range(len(call_list)):
                        if call in match2[idx].at[row,'Lab'] and match2[idx].at[row, call] != match2[idx].at[row, call]:
                            match2[idx].at[row, call] = 'FN'
        return match2
    
    
    def get_confusion_matrix(self, match):
        '''Generates the confusion matrix corresponding to the list of calls gt_indices 
        and the list of predicted calls pred_indices.
        If there are N types of calls, the confusion matrix's size is (N+1)*(N+1);
        the last line and column correspond to the false negatives and false positives respectively,
        in the sense that the call wasn't detected at all (and not merely detected as a different type of call),
        or that the detection doesn't match an actual call.
        '''

        col = self.call_types.copy()
        col.add('FN')
        row = self.call_types.copy()
        cf = pd.DataFrame(columns = col, index = row)
        for call in row:
            for pred in col:
                cf.at[call,pred] = 0
        # The first loop compares the list of calls to all detected calls for every recording.
        for idx in range(len(match)):
            for call in  self.call_types:
                for pred in  self.call_types:
                    if isinstance(match[idx].at[call,pred], list):
                        cf.at[call,pred]+=len(match[idx].at[call,pred])
        # The second loop deals with the FN and FP.
        for idx in range(len(match)):
            for call in self.call_types:
                if isinstance(match[idx].at[call,'FN'], list):
                    cf.at[call,'FN'] += len(match[idx].at[call,'FN'])
        return cf
    
    
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
        cat_frag = pd.DataFrame(columns = self.call_types, index=range(len(label_list)))
        for idx in range(len(cat_frag)):    
            for call in self.call_types:
                cat_frag.at[idx,call] = []
                if isinstance(gt_indices.at[idx,call], list):
                    for call_nb in range(len(gt_indices.at[idx,call])):
                        cat_frag.at[idx,call].append(0)
                        for pred in self.call_types:
                            not_paired = True
                            pred_nb = 0
                            while(not_paired and pred_nb < len(match[idx].at[call,pred])):
                                if match[idx].at[call,pred][pred_nb][0] == call_nb:
                                    not_paired = False
                                    cat_frag.at[idx,call][call_nb] += 1
                                pred_nb += 1
        return cat_frag
                           
                        
    def time_fragmentation(self, match, gt_indices, pred_indices, scale):
        '''How many fragments the call is detected as, indepedently of categories.
        All segments of the call for which one or more prediction have been found 
        is taken into account'''
        time_frag = pd.DataFrame(columns = self.call_types, index = range(len(self.GT_path))) # KD range(len(label_list)))
        idx = 0
        for idx in range(len(gt_indices)):
            print("******************* Performing time fragmentation *******************")
            print(idx)
            for call in self.call_types:
                time_frag.at[idx,call] = []
                if isinstance(gt_indices.at[idx,call], list):
                    for call_nb in range(len(gt_indices.at[idx,call])):
                        time_frag.at[idx,call].append(0)                    
                        matched_pred = []
                        for pred in self.call_types:
                            # print([call, call_nb, pred])
                            for pair_nb in range(len(match[idx].at[call,pred])):
                                if match[idx].at[call,pred][pair_nb][0] == call_nb:
                                    matched_pred.append(pred_indices[pred][idx][match[idx].at[call,pred][pair_nb][1]])
                        matched_pred.sort()
                        if len(matched_pred) > 0:
                            t = 0 # end time of the current predicted call
                            f = 0 # number of fragmetns
                            p = 0 # pair number
                            tmax = max(matched_pred, key = lambda i : i[0])[1] # the latest end of a predicted call that was matched to the GT call
                            while(t < tmax and p < len(matched_pred)):
                                t = matched_pred[p][1]
                                p += 1
                                while(p < len(matched_pred) and matched_pred[p][0] < t):
                                    t = matched_pred[p][1]
                                    p += 1
                                f += 1
                            time_frag.at[idx,call][call_nb] = f
        return time_frag
                    

    def main(self):
        
        # gt_indices = self.get_call_ranges(self.GT_path)
        # pred_indices = self.get_call_ranges(self.pred_path)                          
        gt_indices = self.get_nonfoc_and_foc_calls(self.GT_path)
        if(self.call_analysis in self.call_types):
            for call in self.call_types:
                if call != self.call_analysis:
                    gt_indices[0].loc[:,call] = np.nan
                    gt_indices[1].loc[:,call] = np.nan
        pred_indices = self.get_nonfoc_and_foc_calls(self.pred_path)  
        pred_indices = pred_indices[1]
        
        if(self.call_analysis == "all_types_combined"):
            self.call_types = set(["call","noncall"])
            self.noise_label = "noncall"
            
        for focus in [0,1]:
            if(focus == 0):
                focality = "nonfoc"
            else:
                focality = "foc"            
            match = self.match_prediction_to_labels(gt_indices[focus], pred_indices)
            if(self.call_analysis in self.call_types):
                match2 = self.match_specific_call(gt_indices[focus], pred_indices, match)
            cf = self.get_confusion_matrix(match)
            
            self.call_types.remove(self.noise_label)
            FPos = cf.loc[self.noise_label]
            FPos = FPos.drop(['FN', self.noise_label])
            FNeg = cf['FN']
            FNeg = FNeg.drop([self.noise_label])
            FNeg = pd.DataFrame(FNeg)
            FNeg = FNeg.T
            FNeg.rename(index={'FN': 0}, inplace=True)            
            TPos = dict.fromkeys(self.call_types,0)
            for call in self.call_types:
                TPos[call] = cf.at[call,call]
                for pred in self.call_types:
                    if pred != call:
                        FPos[pred] += cf.at[call,pred]
            TPos = pd.DataFrame(TPos, index=[0])
            FPos = pd.DataFrame(FPos)
            FPos = FPos.transpose()
            FPos.rename(index={self.noise_label: 0}, inplace=True)
            
            Prec = self.precision(TPos,FPos)
            Rec = self.recall(TPos,FNeg)
            Rec2 = self.call_by_call_recall(cf)
            Prec2 = self.new_precision(cf)
            
            offset = self.time_difference(match, gt_indices, pred_indices)
            
            cat_frag = self.category_fragmentation(match, gt_indices, pred_indices)
            
            time_frag = self.time_fragmentation(match, gt_indices, pred_indices, 100)
            
            for i in range(len(time_frag)):
                time_frag.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
                cat_frag.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
                gt_indices[focus].rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
                pred_indices.rename(index={i: self.pred_path[i][94:len(self.pred_path[i])]}, inplace=True)
                
        # for i in range(len(cat_frag)):
        #     pred_indices.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
        # pred_indices.to_csv(os.path.join(metrics_folder, file_name +'_Predictions.csv'))
        
        #KD#
        return Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset

        #KD#
        # precision_filename = 'Precision.csv'
        # recall_filename ='Recall.csv'
        # cat_frag_filename = 'Category_fragmentation.csv'
        # time_frag_filename = 'Time_fragmentation.csv'
        # confusion_filename = 'Confusion_matrix.csv'
        # gt_filename = "Label_indices.csv"
        # pred_filename = "Prection_indices.csv"
        
        # Prec.to_csv( os.path.join(self.results_path, precision_filename))
        # Rec.to_csv( os.path.join(self.results_path, recall_filename))
        # cat_frag.to_csv( os.path.join(self.results_path, cat_frag_filename))
        # time_frag.to_csv(os.path.join(self.results_path, time_frag_filename))
        # cf.to_csv(os.path.join(self.results_path, confusion_filename))
        # gt_indices.to_csv(os.path.join(self.results_path, gt_filename ))
        # pred_indices.to_csv(os.path.join(self.results_path, pred_filename )
        # with open((os.path.join(self.results_path,"Matching_table.txt"), "wb") as fp:   #Pickling
        #     pickle.dump(match, fp)
        # with open((os.path.join(self.results_path,"Time_difference.txt"), "wb") as fp:   #Pickling
        #     pickle.dump(offset, fp)      
        


            


def list_files(directory, ext=".txt"):
    "list_files(directory) - Grab all .txt or specified extension files in specified directory"
    files = glob.glob(os.path.join(directory, "*" + ext))
    files.sort()
    return files




# if __name__=="__main__":
#     # label_dir = "/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/evaluation/evalDuteil/Kiran data/label"
#     label_dir = "/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/saved_models/Kiran model/GT"
#     label_list = list_files(label_dir)


#     #prediction_dir = "/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/evaluation/evalDuteil/Kiran data/prediction"
#     prediction_dir = "/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/saved_models/Kiran model/pred"
#     prediction_list = list_files(prediction_dir)
#     if len(prediction_list) != len(label_list):
#         raise ValueError("Numbers of files don't match")
#     evaluate = Evaluate(label_list, prediction_list, 0.5, 5) # 0.99 is 0.5
#     evaluate.main()