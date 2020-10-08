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
    def __init__(self, label_list, prediction_list, IoU_threshold, gap_threshold): #, results_path
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
        # self.results_path = results_path
        
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
                
    def precision(self, TPos,FPos):
        '''Precision is TP / (TP+FP)'''
        Prec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Prec[call] = TPos.at[0,call] / (TPos.at[0,call] + FPos.at[0,call])
            if(np.isnan(Prec[call])):
                Prec[call] = 1
        Prec = pd.DataFrame(Prec, index=[0])
        return Prec
    
    def recall(self, TPos,FNeg):
        '''Recall is TP / (TP+FN)'''
        Rec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Rec[call] = TPos.at[0,call] / (TPos.at[0,call] + FNeg.at[call])
            if(np.isnan(Rec[call])):
                Rec[call] = 1
        Rec = pd.DataFrame(Rec, index=[0])
        return Rec
    
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
        row.add('FP')
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
                            match[idx].at['FP',pred].append([np.nan,pred_nb])      
                            
        return match
    
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
        row.add('FP')
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
                if isinstance(match[idx].at['FP',call], list):
                    cf.at['FP',call] += len(match[idx].at['FP',call])
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
        cat_frag = pd.DataFrame(columns = self.call_types, index = range(len(self.GT_path))) # KD range(len(label_list)))
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
            print(idx)
            # maxtime = 0
            # for call in self.call_types:
            #     time_frag.at[idx,call] = []
            #     if gt_indices.at[idx,call] == gt_indices.at[idx,call]:
            #         for call_nb in range(len(gt_indices.at[idx,call])):
            #             maxtime = max(gt_indices.at[idx,call][call_nb][1], maxtime)
            #     for pred in self.call_types:
            #         if pred_indices.at[idx,pred] == pred_indices.at[idx,pred]:
            #             for pred_nb in range(len(pred_indices.at[idx,pred])):
            #                 maxtime = max(pred_indices.at[idx,pred][pred_nb][1], maxtime)
            #     maxtime = round(maxtime * scale) # end of last call in the sample in ms
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
                                    # start_time = round(pred_indices[idx][pred][match[idx][call][pred][match_nb][1]][0] * scale)
                                    # end_time = round(pred_indices[idx][pred][match[idx][call][pred][match_nb][1]][1] * scale)
                                    # call_time[start_time:end_time] = 1
                        if len(matched_pred) > 0:
                            t = 0 # end time of the current predicted call
                            f = 0 # number of fragmetns
                            p = 0 # pair number
                            tmax = max(matched_pred, key = lambda i : i[0])[1]
                            while(t < tmax and p < len(matched_pred)):
                                t = matched_pred[p][1]
                                p += 1
                                while(p < len(matched_pred) and matched_pred[p][0] < t):
                                    t = matched_pred[p][1] #p[1]
                                    p += 1
                                f += 1
                            time_frag.at[idx,call][call_nb] = f
                                
                                
                                # end_of_fragment = False
                                # while (not end_of_fragment and f < len(matched_pred)):
                                #     if(matched_pred[f][0] < t):
                                #         t = max(t, matched_pred[f][1])
                                #     else:
                                #         end_of_fragment = True
                                #     f += 1
                                # time_frag.at[idx,call][call_nb] += 1
        return time_frag
                    

    def main(self):
        
        gt_indices = self.get_call_ranges(self.GT_path)
        pred_indices = self.get_call_ranges(self.pred_path)                          

        match = self.match_prediction_to_labels(gt_indices, pred_indices)
        cf = self.get_confusion_matrix(match)
        
        FPos = cf.loc['FP']
        FPos = FPos.drop(['FN'])
        FNeg = cf['FN']
        FNeg = FNeg.drop(['FP'])
        TPos = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            TPos[call] = cf.at[call,call]
            for pred in self.call_types:
                if pred != call:
                    FPos[pred] += cf.at[call,pred]
        TPos = pd.DataFrame(TPos, index=[0])
        FPos = pd.DataFrame(FPos)
        FPos = FPos.transpose()
        FPos.rename(index={'FP': 0}, inplace=True)
        
        
        # TPos = pd.Series(columns = [0], index = self.call_types)
        # FPos = pd.DataFrame(columns = [0], index = self.call_types)
        # for call in self.call_types:
        #     TPos[call,0] = cf.at[call,call]
        #     FPos.at[call,0] = 0
        #     for pred in self.call_types:
        #         if pred != call:
        #             FPos.at[call,0] += cf.at[call,pred]
        
        Prec = self.precision(TPos,FPos)
        Rec = self.recall(TPos,FNeg)
        
        offset = self.time_difference(match, gt_indices, pred_indices)
        
        cat_frag = self.category_fragmentation(match, gt_indices, pred_indices)
        
        time_frag = self.time_fragmentation(match, gt_indices, pred_indices, 100)
        
        for i in range(len(time_frag)):
            time_frag.rename(index={i: self.pred_path[i][94:len(self.pred_path[i])]}, inplace=True)
            cat_frag.rename(index={i: self.pred_path[i][94:len(self.pred_path[i])]}, inplace=True)
            gt_indices.rename(index={i: self.pred_path[i][94:len(self.pred_path[i])]}, inplace=True)
            pred_indices.rename(index={i: self.pred_path[i][94:len(self.pred_path[i])]}, inplace=True)
            #KD#
            # time_frag.rename(index={i: prediction_list[i][94:len(prediction_list[i])]}, inplace=True)
            # cat_frag.rename(index={i: prediction_list[i][94:len(prediction_list[i])]}, inplace=True)
            # gt_indices.rename(index={i: prediction_list[i][94:len(prediction_list[i])]}, inplace=True)
            # pred_indices.rename(index={i: prediction_list[i][94:len(prediction_list[i])]}, inplace=True)
        
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
        
        # print("Done!")
        


            


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