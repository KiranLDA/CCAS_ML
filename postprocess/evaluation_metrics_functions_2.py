
import numpy as np
import glob
import os
import tensorflow as tf
import pandas as pd
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)
# newer version
# another comment

class Evaluate:
    def __init__(self, label_list, prediction_list, model, run, noise_label = "noise", IoU_threshold = 0.5, low_thresh = "", high_thresh = "", call_analysis = "normal", GT_proportion_cut = 0):
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
        
        all_files = [label_list, prediction_list]
        self.headers = set(['Label', 'Duration', 'Start', 'End'])
        
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
                if(len(self.headers.intersection(set(column_names))) != len(self.headers)):
                    raise ValueError("File %s missing headers %s"%(file, self.headers - set(column_names))) # checks that all the headers are there
        
        self.GT_path = sorted(label_list)
        self.pred_path = sorted(prediction_list)
        self.IoU_threshold = IoU_threshold
        if noise_label in self.call_types:
            self.noise_label = noise_label
        else:
            raise ValueError("Unknown noise label %s"%(noise_label))
        self.no_call = set(["noise", "beep", "synch"])
        self.true_call= set(self.call_types.difference(self.no_call))
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        
        self.call_analysis = call_analysis
        self.model = model
        self.run = run
        self.min_call_length = {}
        # self.min_call_length = {"agg":7, "al":9, "cc":16, "ld":24, "mo":15, "sn":5, "soc":10}
        self.frame_rate = 200
        self.GT_proportion_cut = GT_proportion_cut
        
        
    def get_call_ranges(self, tablenames, data_type):
        '''
        This function takes the list of prediction or label 'files tablenames' 
        and format its data as a dataframe, with rows representing the data
        file, the columns representing the call types and each cell containing
        an ordered list of the beginning and end time of the different calls or
        predictions for the corresponding file and call type.
        The output is that table of calls, as well as a table of which is focal
        or non-focal.
        '''
        
        nonfoc_tags = ["NONFOC", "nf", "*"]  # presence of any of these strings in the Label column indicates a non-focal call (in the third case, a call that is ambiguous). This is only possible in the ground truth.   
        skipped = 0
        calls_indices = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))
        non_foc_gt = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))
        
        #Determine the minimum length authorised for a prediction (so 0 for the GT, any length is authorised)
        all_min_call_length = {}
        if data_type == "pred":
            for call in self.call_types:
                if call in self.min_call_length.keys():
                    all_min_call_length[call] = self.min_call_length[call]
                else:
                    all_min_call_length[call] = 0
        else:
            for call in self.call_types:
                all_min_call_length[call] = 0
            
        
        for c in self.call_types: # All dataframes are initialised with empty lists to avoid dealing with NaN's and lists separately.
            calls_indices[c] = [[] for f in range(len(calls_indices))]
            non_foc_gt[c] = [[] for f in range(len(non_foc_gt))]
        
        calls = list(self.call_types)
        calls.sort()
        total_examples = 0
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
                if table.Label[row] in ['skipon', 'SKIPON']: # parts of some files must be skipped. Those are surrounded by two rows with the labels 'skipon' and 'skipoff'.
                    while(table.Label[row] not in  ['skipoff', 'SKIPOFF'] and row < len(table) and not table_end):
                        row += 1
                        skipped += 1
                else:
                    if table.Label[row] in ['END', 'STOP']:
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
                        # the beginning and end times for that call are added to the list 
                        if table.Duration[row] > all_min_call_length[actual_call]: # / (self.frame_rate - 1):
                            calls_indices[actual_call][i].append((table.Start[row], table.End[row]))
                            if any(word in table.Label[row] for word in nonfoc_tags):
                                non_foc_gt[actual_call][i].append(True)
                            else:
                                non_foc_gt[actual_call][i].append(False)
                            extract +=1
                    row += 1
            total_examples += len(table)
            
            # Everything after the end is skipped
            while(row < len(table)):
                skipped += 1
                row += 1  
            print(str(skipped) + " out of " + str(len(table)) + " entries were skipped.")
            
            if(self.call_analysis == "all_types_combined"): # converts the specific call types into call/noncalls
                calls_combined = pd.DataFrame(columns = ["call", "noncall"], index = range(len(calls_indices)))
                for call in self.call_types:
                    if isinstance(calls_indices.at[i,call], list):
                        if call in self.true_call:
                            if isinstance(calls_combined.at[i,"call"], list):
                                calls_combined.at[i,"call"]  = calls_combined.at[i,"call"] + calls_indices.at[i,call]
                            else:
                                calls_combined.at[i,"call"] = calls_indices.at[i,call]
                        else:
                            if isinstance(calls_combined.at[i,"noncall"], list):
                                calls_combined.at[i,"noncall"]  = calls_combined.at[i,"noncall"] + calls_indices.at[i,call]
                            else:
                                calls_combined.at[i,"noncall"] = calls_indices.at[i,call]
                    calls_indices = calls_combined
        return calls_indices, non_foc_gt
    
    
    def get_min_call_length(self, gt_indices):
        for call in self.true_call:
            if call not in self.min_call_length.keys():
                duration = []
                for idx in range(len(gt_indices)):
                    for call_num in range(len(gt_indices.at[idx,call])):
                        duration.append(gt_indices.at[idx,call][call_num][1] - gt_indices.at[idx,call][call_num][0])
                duration.sort()
                shortest_call_allowed = int(len(duration) * self.GT_proportion_cut) 
                self.min_call_length[call] = duration[shortest_call_allowed]
                
    
    
    def get_nonfoc_and_foc_calls(self, tablenames):
        ''' Like get_call_ranges, but separates focal and nonfocal calls in two
        different tables. The non-focal calls are marked as such in the label 
        tables. There is no distinction of focal and non-focal for prediction
        files.'''
        
        nonfoc_tags = ["NONFOC", "nf", "*"]    
        skipped = 0
        
        foc = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))         # table that sorts the focal calls by file and call type
        nonfoc = pd.DataFrame(columns = self.call_types, index=range(len(tablenames)))      # table that sorts the non-focal calls by file and call type
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
                if table.Label[row] in ['skipon', 'SKIPON']: # parts of some files must be skipped. Those are surrounded by two rows with the labels 'skipon' and 'skipoff'.
                    while(table.Label[row] not in  ['skipoff', 'SKIPOFF'] and row < len(table) and not table_end):
                        row += 1
                        skipped += 1
                else:
                    if table.Label[row] in ['END', 'STOP']:
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
                        # the beginning and end times for that call are added to the appropriate list (or the list is created if this is the first call of that type in the file), depending on whether a nonfocal tag is present in the label.
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
            
            if(self.call_analysis == "all_types_combined"): # converts the specific call types into call/noncalls
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

        return nonfocfoc  # x, y
                
    
    def precision(self, TPos,FPos):
        '''Precision is TP / (TP+FP)'''
        Prec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Prec[call] = TPos.at[0,call] / (TPos.at[0,call] + FPos.at[0,call])
            if(np.isnan(Prec[call])): # If the call was never predicted, the precision is set to 1.0 by convention.
                Prec[call] = 1.0
        Prec = pd.DataFrame(Prec, index=[0])
        return Prec
    
    def lenient_precision(self, cm):
        ''' Like precision, but a match is considered a true positive as long 
        as the call is matched to an actual call type. This metric allows us to
        determine whether animal sounds are properly classified as such.'''
        Prec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            all_preds = sum(cm[call])
            if all_preds == 0:
                Prec2[call] = np.nan
            else:
                Prec2[call] = (all_preds - cm.at[self.noise_label,call]) / all_preds
            if(np.isnan(Prec2[call])): # If the call was never predicted, the precision is set to 1.0 by convention.
                Prec2[call] = 1.0
        Prec2 = pd.DataFrame(Prec2, index=[0])
        return Prec2
    
    def recall(self, TPos,FNeg):
        '''Recall is TP / (TP+FN)'''
        Rec = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            Rec[call] = TPos.at[0,call] / (TPos.at[0,call] + FNeg.at[0,call])
            if(np.isnan(Rec[call])): # If the call is not present in the labelled data, the precision is set to 1.0 by convention.
                Rec[call] = 1.0 
        Rec = pd.DataFrame(Rec, index=[0])
        return Rec
    
    def lenient_recall(self, cm):
        ''' Like recall, but a match is considered a true positive as long 
        as the call is matched to an actual call type. This metric allows us to
        determine whether animal sounds are properly classified as such.'''        
        Rec2 = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            GT = sum(cm.loc[call])
            if GT == 0:
                Rec2[call] = np.nan
            else:
                Rec2[call] = (GT - cm.at[call, 'FN']) / GT
            if(np.isnan(Rec2[call])): # If the call is not present in the labelled data, the precision is set to 1.0 by convention.
                Rec2[call] = 1.0 
        Rec2 = pd.DataFrame(Rec2, index=[0])
        return Rec2
    
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
    
    def match_prediction_to_labels(self, gt_indices, pred_indices):
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
                                    if better_paired: # Pariing for the strict match.
                                        if pred_end - pred_start > min_length: #/ (self.frame_rate - 1): # checking that the length of the prediction is reasonable
                                            paired_pred.at[idx,pred][pred_nb] = True
                                            paired_call.at[idx,pred][call_nb] = True   
                                            match[idx].at[pred,pred].append((call_nb,pred_nb))
        
        # Finding the wrong detections and false negatives:
        for idx in range(np.size(gt_indices,0)):
            print(idx)
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
                                                if better_paired:
                                                    if pred_end - pred_start > min_length: # / (self.frame_rate - 1): # checking that the lenght of the prediction is reasonable
                                                        paired_pred.at[idx,pred][pred_nb] = True
                                                        matched_call = True
                                                        match[idx].at[call,pred].append((call_nb,pred_nb))                                                
                                                
                        if not matched_call:
                            match[idx].at[call,'FN'].append((call_nb,np.nan))
                            loose_match[idx].at[call,'FN'].append((call_nb,np.nan)) 
                                                
        # At this point all labelled calls have been matched or classified as false negatives.
        # Only the false positives still need to be marked.        
        for idx in range(np.size(pred_indices,0)):
            for pred in self.call_types:
                if isinstance(pred_indices.at[idx,pred], list):
                    for pred_nb in range(len(pred_indices.at[idx,pred])):
                        if not paired_pred.at[idx,pred][pred_nb]:
                            match[idx].at[self.noise_label,pred].append((np.nan,pred_nb))  
                            loose_match[idx].at[self.noise_label,pred].append((np.nan,pred_nb))  
                            # FP are sorted in noise. One consequence of that is that the call type noise 
                            # can be properly identified and still increase the number of false positives.
                            
        return match, loose_match
    
    
    def match_specific_call(self, gt_indices, pred_indices, match, non_foc_gt, non_foc_pred):
        '''For a given call type, finds all the prediction associated to the matches
        and vice versa. Mostly corresponds to merging the different cells in a match
        table, except this account for doubles. This function returns, for a given 
        call type, the list of matches for the labelled calls, and the list of 
        matches for the predictions.'''
        call_match = []
        pred_match = []
        while len(call_match) < np.size(gt_indices,0):
            call_match.append([])
            pred_match.append([])
        # So predictions are matched to the correct call by default, the list of call types must end with the correct call.
        call_list = self.true_call.copy()
        call_list.remove(self.call_analysis)
        call_list = list(call_list) + [self.call_analysis] 
        
        # 
        for idx in range(np.size(gt_indices,0)):
            call_match[idx] = [np.nan] * len(gt_indices.at[idx,self.call_analysis])
            for pred in call_list:  
                if pred in self.min_call_length.keys():
                    min_length = self.min_call_length[pred]
                else:
                    min_length = 0
                if isinstance(match[idx].at[self.call_analysis,pred],list):
                    for match_nb in range(len(match[idx].at[self.call_analysis,pred])):
                        duration = pred_indices.at[idx, pred][match[idx].at[self.call_analysis,pred][match_nb][1]][1] - pred_indices.at[idx, pred][match[idx].at[self.call_analysis,pred][match_nb][1]][0]
                        if duration > min_length  and not non_foc_pred.at[idx,pred][match[idx].at[self.call_analysis,pred][match_nb][1]]:
                            call_match[idx][match[idx].at[self.call_analysis,pred][match_nb][0]] = (pred, match[idx].at[self.call_analysis,pred][match_nb][1])
        
        #                     
        for idx in range(np.size(gt_indices,0)):
            pred_match[idx] = [np.nan] * len(pred_indices.at[idx,self.call_analysis])
            for call in call_list:
                if call in self.min_call_length.keys():
                    min_length = self.min_call_length[call]
                else:
                    min_length = 0                    
                if isinstance(match[idx].at[call,self.call_analysis],list):
                    for match_nb in range(len(match[idx].at[call,self.call_analysis])):
                        # if call == "oth":
                        # print([idx, call, match_nb])
                        duration = gt_indices.at[idx, call][match[idx].at[call,self.call_analysis][match_nb][0]][1] - gt_indices.at[idx, call][match[idx].at[call,self.call_analysis][match_nb][0]][0]
                        if duration > min_length  and not non_foc_gt.at[idx,call][match[idx].at[call,self.call_analysis][match_nb][0]]:
                            pred_match[idx][match[idx].at[call,self.call_analysis][match_nb][1]] = (call, match[idx].at[call,self.call_analysis][match_nb][0])        

        return call_match, pred_match
                        
    
    def non_focal_prediction(self, pred_indices, non_foc_gt, match):
        # Creates a table of which predictions are associated with non-focal recordings
        non_foc_pred = pd.DataFrame(columns = self.call_types, index=range(len(pred_indices)))
        for idx in range(np.size(pred_indices,0)):
            for call in self.call_types:
                non_foc_pred.at[idx,call] = [False]*len(pred_indices.at[idx,call])
                # Filling non_foc_pred
        for idx in range(np.size(non_foc_gt,0)):
            for call in self.call_types:
                for pred in self.call_types:
                    for call_num in range(len(match[idx].at[call,pred])):
                        if not np.isnan(match[idx].at[call,pred][call_num][0]) and non_foc_gt.at[idx,call][match[idx].at[call,pred][call_num][0]]:
                            non_foc_pred.at[idx,pred][match[idx].at[call,pred][call_num][1]] = True
        return non_foc_pred
                    
    
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
            if self.call_analysis in self.call_types:
                possible_calls = [self.call_analysis]
            else:
                possible_calls = self.call_types
            for call in possible_calls:
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
        time_frag = pd.DataFrame(columns = self.call_types, index=range(len(gt_indices)))
        idx = 0
        for idx in range(len(gt_indices)):
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
                            time_frag.at[idx,call][call_nb] = f
        return time_frag
    

            
    def evaluation(self, gt_indices, pred_indices, focus, non_foc_gt):
        # Matches the labelled data with the predicted calls. loose_match is equivalent to a match with an intersection of union threshold of 0.
        match, loose_match = self.match_prediction_to_labels(gt_indices, pred_indices)
        if(self.call_analysis in self.call_types):
            for call in self.call_types:
                if call != self.call_analysis:
                    # For a one call analysis, the labelled data for the other calls are discarded after the matching so false positives can be assessed accurately.
                    gt_indices.loc[:,call] = np.nan    
        if self.call_analysis == "call_type_by_call_type":
            true_calls = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc'}
            call_match = pd.DataFrame(columns = true_calls, index=range(len(gt_indices)))
            pred_match = pd.DataFrame(columns = true_calls, index=range(len(gt_indices)))
            non_foc_pred = self.non_focal_prediction(pred_indices, non_foc_gt, match)
                
            for CALL in true_calls:
                self.call_analysis = CALL
                call_match[CALL], pred_match[CALL] = self.match_specific_call(gt_indices, pred_indices, match, non_foc_gt, non_foc_pred)
            self.call_analysis = "call_type_by_call_type"


        if self.call_analysis in self.true_call:
            match2 = self.match_specific_call(gt_indices, pred_indices, match, non_foc_gt, non_foc_pred)
        cm = self.get_confusion_matrix(match)
        self.call_types.remove(self.noise_label) # As the noise is also the label for the false positives, it would cause problems to leave it in the data at this point.
        FPos = cm.loc[self.noise_label]
        FPos = FPos.drop(['FN', self.noise_label])
        FNeg = cm['FN']
        FNeg = FNeg.drop([self.noise_label])
        FNeg = pd.DataFrame(FNeg)
        FNeg = FNeg.T
        FNeg.rename(index={'FN': 0}, inplace=True)
        TPos = dict.fromkeys(self.call_types,0)
        for call in self.call_types:
            TPos[call] = cm.at[call,call]
            for pred in self.call_types:
                if pred != call: 
                    FPos[pred] += cm.at[call,pred]
        TPos = pd.DataFrame(TPos, index=[0])
        FPos = pd.DataFrame(FPos)
        FPos = FPos.transpose()
        FPos.rename(index={self.noise_label: 0}, inplace=True)   
        Prec = self.precision(TPos,FPos)
        Rec = self.recall(TPos,FNeg)
        lenient_Rec = self.lenient_recall(cm)
        lenient_Prec = self.lenient_precision(cm)
        offset = self.time_difference(match, gt_indices, pred_indices)
        cat_frag = self.category_fragmentation(loose_match, gt_indices, pred_indices)
        time_frag = self.time_fragmentation(loose_match, gt_indices, pred_indices, 100)
        
        # preparing the metrics for saving
        for i in range(len(gt_indices)):
            time_frag.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
            cat_frag.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
            gt_indices.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
            pred_indices.rename(index={i: os.path.basename(prediction_list[i])}, inplace=True)
        
        # Creating the metrics folder
        # main_dir =  os.path.join("/media/mathieu/Elements/code/KiranLDA/results/", self.model, self.run, "metrics")
        # metrics_folder = os.path.join(main_dir, self.call_analysis, str(self.GT_proportion_cut), str(self.low_thresh), str(self.high_thresh))
        # directories = [main_dir,
        #                 os.path.join(main_dir, self.call_analysis),
        #                 os.path.join(main_dir, self.call_analysis, str(self.GT_proportion_cut)),
        #                 os.path.join(main_dir, self.call_analysis, str(self.GT_proportion_cut), str(self.low_thresh)),
        #                 metrics_folder]
        # for diri in directories:
        #     if not os.path.exists(diri):
        #         os.mkdir(diri)                
        
        # Saving the metrics
        # Prec.to_csv(os.path.join(metrics_folder, focus +'_Precision.csv'))
        # lenient_Prec.to_csv(os.path.join(metrics_folder, focus +'_LenientPrecision.csv'))
        # Rec.to_csv(os.path.join(metrics_folder, focus +'_Recall.csv'))
        # lenient_Rec.to_csv(os.path.join(metrics_folder, focus +'_LenientRecall.csv'))
        # cat_frag.to_csv(os.path.join(metrics_folder, focus +'_Category fragmentation.csv'))
        # time_frag.to_csv(os.path.join(metrics_folder, focus +'_Time fragmentation.csv'))
        # cm.to_csv(os.path.join(metrics_folder, focus +'_Confusion matrix.csv'))
        # with open(os.path.join(metrics_folder, focus +'_Ground truth.p'), 'wb') as fp:
        #     pickle.dump(gt_indices, fp)
        # with open(os.path.join(metrics_folder, focus +'_Predictions.p'), 'wb') as fp:
        #     pickle.dump(pred_indices, fp)
        # with open(os.path.join(metrics_folder, focus +'_Matching table.txt'), 'wb') as fp:
        #     pickle.dump(match, fp)
        # with open(os.path.join(metrics_folder, focus +'_Time difference.txt'), 'wb') as fp:
        #     pickle.dump(offset, fp)  
        # if self.call_analysis == "call_type_by_call_type":
        #     with open(os.path.join(metrics_folder, focus +'_call match.p'), 'wb') as fp:
        #         pickle.dump(call_match, fp)
        #     with open(os.path.join(metrics_folder, focus +'_pred match.p'), 'wb') as fp:
        #         pickle.dump(pred_match, fp)
        # if self.call_analysis in self.true_call:
        #     match2.to_csv(os.path.join(metrics_folder, focus + self.call_analysis + ' match.csv'))
            
            
        return Prec, lenient_Prec, Rec, lenient_Rec, cat_frag, time_frag, cm, gt_indices, pred_indices, match, offset, call_match, pred_match, match2

    def main(self):
        
        # Generating tables of calls and predictions.
        if(self.call_analysis == "normal"): 
            gt_indices = self.get_nonfoc_and_foc_calls(self.GT_path)
            pred_indices = self.get_nonfoc_and_foc_calls(self.pred_path)  
            pred_indices = pred_indices[1] # at this stage there is no disctinction of focal or non-focal calls, so the empty non-focal prediction table is removed. 
        else:            
            gt_indices, non_foc_gt = self.get_call_ranges(self.GT_path, "GT")
            self.get_min_call_length(gt_indices)
            pred_indices, _ = self.get_call_ranges(self.pred_path, "pred") 
        
        if(self.call_analysis == "all_types_combined"):
            # reducing the call types to a binary classification.
            self.call_types = set(["call","noncall"])
            self.noise_label = "noncall"
        
        if(self.call_analysis == "normal"): 
            for foc in [0,1]:
                if(foc == 0):
                    focus = "nonfoc"
                else:
                    focus = "foc"
                self.evaluation(gt_indices[foc], pred_indices, focus, [])
        else:
            self.evaluation(gt_indices, pred_indices, "", non_foc_gt)      




def list_files(directory, ext=".txt"):
    "list_files(directory) - Grab all .txt or specified extension files in specified directory"
    files = glob.glob(os.path.join(directory, "*" + ext))
    files.sort()
    return files


if __name__=="__main__":
    model = "NoiseAugmented_ProportionallyWeighted_NoOther_2020-10-14_03:12:32.817594"
    run = "new_run"
    main_dir =  os.path.join("/media/mathieu/Elements/code/KiranLDA/results", model, run)
    label_dir = os.path.join(main_dir, "label_table")
    label_list = list_files(label_dir)
    
    ''' The model makes a prediction for any time point in the wav file to be 
    associated with a given type call. When a time point reaches a prediction 
    score higher than an upper threshold (thresh), every point in the wav file 
    before and after it is considered as belonging to a predicted call, until
    a point for which the score is below a lower threshold (low_thresh). Thus 
    the closer the two thresholds are, the more calls will be predicted; the 
    lower the upper threshold, the more false positives; the higher the upper 
    threshold, the more false negatives. Such predictions have been run for 
    various combinations of lower and higher thresholds. This program computes
    various evaluation metrics for each set of predictions.
    '''
    for GT_proportion_cut in [0,0.005,0.01,0.015,0.02,0.025,0.03]:
        for low_thresh in [0.2,0.3]:
            for thresh in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]:
                if low_thresh < thresh:
                    print([GT_proportion_cut, low_thresh, thresh])
                    
                    '''There are eleven classes of meerkat calls, these seven are 
                    the calls we're interested in.There are also two classes that
                    corresponds to artefacts of the recording method ('beep' and 
                    'synch'), a noise class, and an other class ('oth') for the 
                    calls that couldn't be properly labelled'''
                    
                    true_calls = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc'}
                    
                    results_dir = os.path.join(main_dir, "results_per_threshold", str(low_thresh), str(thresh))
                    prediction_list = list_files(results_dir) #list of the paths where the prediction are stored, each file corresponds to one recording session for one animal.
                    
                    if len(prediction_list) != len(label_list):
                        raise ValueError("Numbers of files don't match")
                    # for CALL in true_calls:
                        # evaluate = Evaluate(label_list, prediction_list, noise_label = "noise", IoU_threshold = 0.5, gap_threshold = 5, high_thresh = thresh, call_analysis = "oth")
                    evaluate = Evaluate(label_list, prediction_list, model = model, run = run, low_thresh = low_thresh, high_thresh = thresh, call_analysis = "call_type_by_call_type", GT_proportion_cut = GT_proportion_cut)
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
        


