''' draws the distributions of prediction scores. 
All the predictions are centred around their maximum.'''

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob
import pickle
import random

def plotting(data, classification, low_thresh, thr, call):   
    plt.figure()
    plt.title("Score profiles for " + classification + " " + call + " calls, thresholds = " + str(low_thresh) + " / " + str(thr))
    plt.xlabel('time')
    plt.ylabel('score')
    mint0 = 1
    maxt1 = -1
    for profile in data:
        max_value = max(profile)
        ind_max = profile.index(max_value)
        t0 = -ind_max
        mint0 = min(mint0, t0)
        t1 = len(profile) - ind_max
        maxt1 = max(maxt1, t1)
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        plt.plot(range(t0,t1), profile, c=color)
    plt.axis([mint0, maxt1, 0, 1])
    plt.savefig('/tmp/' + classification + f'-{low_thresh}-{thr}.png' ) 

model = "NoiseAugmented_ProportionallyWeighted_NoOther_2020-10-14_03:12:32.817594"
run = "new_run"
main_dir =  os.path.join("/media/mathieu/Elements/code/KiranLDA/results", model, run)
low_thresh = 0.2
all_thr = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
call_types = {'agg', 'al', 'beep', 'cc', 'ld', 'mo', 'noise', 'oth', 'sn', 'soc', 'synch'}
true_calls = {'agg', 'al', 'cc', 'ld', 'mo', 'sn', 'soc'}
non_calls = {'beep', 'noise', 'oth', 'synch'}

for thr in all_thr:
    if thr > low_thresh:
        print(str(thr))
        # Generating the prediction tables
        thr_path = os.path.join(main_dir, "metrics", "call_type_by_call_type", str(low_thresh), str(thr))
        pred_match = pickle.load(open(os.path.join(thr_path, "_pred match.p"), "rb"))
        pred_path = os.path.join(main_dir,"predictions", "pred_table", str(low_thresh), str(thr))
        filenames = sorted(glob.glob(os.path.join(pred_path, "*.txt")))
        pred_indices = pd.DataFrame(columns = call_types, index=range(len(filenames)))
        TP_scores = pd.DataFrame(index=range(len(filenames)), columns = call_types)
        FP_scores = pd.DataFrame(index=range(len(filenames)), columns = call_types)
        misclassified_scores = pd.DataFrame(index=range(len(filenames)), columns = call_types)
        non_calls_scores = pd.DataFrame(index=range(len(filenames)), columns = call_types)
        faulty_call_files = pd.DataFrame(False, index=range(len(filenames)), columns = call_types)
        for c in call_types:
            TP_scores[c] = [[] for f in filenames]
            FP_scores[c] = [[] for f in filenames]
            misclassified_scores[c] = [[] for f in filenames]
            non_calls_scores[c] = [[] for f in filenames]
            
        for i in range(len(filenames)):
            table = pd.read_csv(filenames[i], delimiter=';') 
            row = 0
            # go to Start
            while(row < len(table) and not table.Label[row] in ['START','start']):
                row += 1
            # main loop
            table_end = False # whether we've reached the 'End' label
            while(row < len(table) and not table_end):
                if table.Label[row] in ['skipon', 'SKIPON']:
                    while(table.Label[row] not in  ['skipoff', 'SKIPOFF'] and row < len(table) and not table_end):
                        row += 1
                else:
                    if table.Label[row] in ['END', 'STOP']:
                        table_end = True
                    actual_call = None
                    to_be_skipped = False # There should be one 'True' per line. In any other case, the line will be skipped and counted as such.
                    for call in call_types:
                        if (table.at[row,call]):  
                            if(actual_call is None):
                                actual_call = call
                            else:
                                to_be_skipped = True
                    if(actual_call is None):
                        print("error")
                    else:
                        if(pred_indices[actual_call][i] != pred_indices[actual_call][i]):
                            pred_indices[actual_call][i] = [(table.Start[row], table.End[row])]
                        else:
                            pred_indices[actual_call][i].append((table.Start[row], table.End[row]))
                            
                        # entering the list of scores in the score table:
                        if isinstance(table.at[row,"scores"],str):
                            string_series = table.at[row,"scores"][1:-1]
                            float_scores = [float(item) for item in string_series.split(", ")]                        
                            if actual_call in non_calls:
                                (non_calls_scores.at[i, actual_call]).append(float_scores)
                            else:
                                # print(thr, i, row, actual_call)
                                try:
                                    prediction = pred_match.at[i,actual_call][len(pred_indices[actual_call][i])-1]
                                    if isinstance(prediction, tuple):
                                        if prediction[0] == actual_call:
                                            (TP_scores.at[i, actual_call]).append(float_scores)
                                        else:
                                            (FP_scores.at[i, actual_call]).append(float_scores)
                                    else: 
                                        (misclassified_scores.at[i, actual_call]).append(float_scores)
                                except:
                                    faulty_call_files.at[i,call] = True
                                    print("don't include the calls of type " + actual_call + " from file " + str(i) + " in the analysis (" + filenames[i] + ")")
                    row += 1
                
        for call in call_types:
            all_TP = []
            all_FP = []
            all_misclassified = []
            all_non_calls = []
            for i in range(len(filenames)):
                if not faulty_call_files.at[i,call]:
                    if len(TP_scores.at[i,call]) > 0:
                        all_TP += TP_scores.at[i,call]
                    if len(FP_scores.at[i,call]) > 0:
                        all_FP += FP_scores.at[i,call]
                    if len(misclassified_scores.at[i,call]) > 0:
                        all_misclassified += misclassified_scores.at[i,call]
                    if len(FP_scores.at[i,call]) > 0:
                        all_non_calls += non_calls_scores.at[i,call]
            if len(all_TP) > 0:
                plotting(all_TP, "true positives", low_thresh, thr, call)
            if len(all_FP) > 0:    
                plotting(all_FP, "false positives", low_thresh, thr, call)
            if len(all_misclassified) > 0:
                plotting(all_misclassified, "misclassified", low_thresh, thr, call)
            if len(all_non_calls) > 0:
                plotting(all_non_calls, "non", low_thresh, thr, call)