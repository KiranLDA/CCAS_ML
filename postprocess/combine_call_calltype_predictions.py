#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 17:27:29 2021

@author: kiran
"""
import pandas as pd
import numpy as np
import ast
github_dir = "/home/kiran/Documents/github/CCAS_ML"

# add path to local functions
import os
os.chdir(github_dir)
import postprocess.evaluation_metrics_functions as metrics
eval_analysis = "call_type_by_call_type"

import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.colors import LogNorm

call_len_dict = dict()
for i in call_types.keys():
    call_len_dict[i] = sum(label_table[i].astype(int))
    
for low_thr_1 in [0.01,0.05,0.1,0.2]:
    for high_thr_1 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]: 
        for low_thr_2 in [0.01,0.05,0.1,0.2]:
            for high_thr_2 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]: 
                if high_thr_1>low_thr_1:
                    if high_thr_2>low_thr_2:

                        calltype_name = "/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_CALLTYPE_PRED_TABLE_thr_"+ str(low_thr_1) +"-"+ str(high_thr_1 )+".txt"
                        callpresence_name = "/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_CALLPRESENCE_PRED_TABLE_thr_"+ str(low_thr_2) +"-"+ str(high_thr_2) +".txt"               
                        
                        calltype_table = pd.read_csv(calltype_name, 
                                                     delimiter=';') 
                        callpresence_table = pd.read_csv(callpresence_name,
                                                         delimiter=';') 
                        label_table = pd.read_csv("/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/label_table//HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_LABEL_TABLE.txt",
                                                         delimiter=';') 
                        
                        
                        
                        calltype_table["max_score"] = calltype_table["scores"].apply(lambda x: ast.literal_eval(x) if (len(x) == 3 ) else max(ast.literal_eval(x)))
                        calltype_table["median_score"] = calltype_table["scores"].apply(lambda x: ast.literal_eval(x) if (len(x) == 3 ) else np.median(ast.literal_eval(x)))
                        
                        #calltype_table["max_score"].plot.hist(bins=100)
                        #calltype_table["median_score "].plot.hist(bins=100)
                        
                        callpresence_table["calltype"] = "Not matched"
                        
                        
                        pred_table = pd.DataFrame()
                        # pred_table = pred_table.append(calltype_table.iloc[0])
                        for row in range(len(callpresence_table["Start"])):
                            # if callpresence_table["noise"].iloc[row]:
                            #     pred_table = pred_table.append(callpresence_table.iloc[row])
                            # else:
                            overlap = list((
                                (((calltype_table['Start']>= callpresence_table["Start"].iloc[row]) & (calltype_table['End'] <= callpresence_table["End"].iloc[row]))|
                                    ((calltype_table['End']>= callpresence_table["Start"].iloc[row]) & (calltype_table['End'] <= callpresence_table["End"].iloc[row]))|
                                    ((calltype_table['Start']<= callpresence_table["Start"].iloc[row]) & (calltype_table['End'] >= callpresence_table["End"].iloc[row])) |
                                    ((calltype_table['Start']>= callpresence_table["Start"].iloc[row]) & (calltype_table['Start'] <= callpresence_table["End"].iloc[row])))
                                & ((calltype_table['End'] - calltype_table['Start']) <= max(label_table["Duration"])))
                                )
                            overlap_idx = [i for i, e in enumerate(overlap) if e != 0]
                            if len(overlap_idx) != 0 :
                                overlap_medians = [calltype_table["median_score"].iloc[i] for i in overlap_idx]
                                max_call_idx = overlap_idx[overlap_medians.index(max(overlap_medians))]
                                callpresence_table["calltype"].iloc[row] = calltype_table["Label"].iloc[max_call_idx]
                                pred_table = pred_table.append(calltype_table.iloc[max_call_idx])
                        
                        
                        max(label_table["Duration"])
                        
                        
                        pred_table = pred_table.sort_values(by=['Start'])
                        pred_table = pred_table.reset_index()
                        del pred_table['index']
                        # del label_table['index']
                        # del label_table['level_0']
                        del pred_table['max_score']
                        del pred_table['median_score']
                        
                        for label in call_types:
                            pred_table[label] = (pred_table[label]).astype(bool)
                        
                        
                        
                        
                        pred_table.to_csv("/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_test_combined.txt", 
                                                               index=None, sep=';', mode = 'w')
                        # pred_table = pd.read_csv("/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_test_combined.txt", 
                        #                              delimiter=';') 
                        
                        ######################################################################

                        
                        label_list =  ["/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/label_table//HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_LABEL_TABLE.txt"]
                        pred_list = ["/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_test_combined.txt"]
                        # pred_list = ["/media/kiran/D0-P1/animal_data/meerkat/EXAMPLE_NoiseAugmented_0.3_0.8_NotWeighted_MaskedOther_Forked/test_data/predictions/pred_table/HM_VHMM021_MBLT_R01_20190707-20190719_file_12_(2019_07_18-11_44_59)_185944_CALLTYPE_PRED_TABLE_thr_0.01-0.1.txt"]
                        
                        
                        
                        call_types = {
                            'cc' :["cc","Marker", "Marque"],
                            'sn' :["sn","subm", "short","^s$", "s ", "s\*"],
                            'mo' :["mo","MOV","MOVE"],
                            'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
                            'ld' :["ld","LD","lead","LEAD"],
                            'soc':["soc","SOCIAL", "so ", "so"],
                            'al' :["al "," al ", " al","ALARM", "^al$"],
                            'beep':["beep", "beeb"],
                            'synch':["sync"],
                            'oth':["oth","other","lc", "lost","hyb","HYBRID","fu","sq", "seq","\+","ukn","unk","unknown",  "\#","\?"],
                                    #unsure calls
                                    # "x", "\%","\*", #noisy callsa
                                    #"\$",
                            'noise':['start','stop','end','skip']
                            }
                        
                        
                        # testing_label_dict = dict()
                        # for label in call_types: 
                        #     testing_label_dict[label] = 0
                        # testing_label_dict["noise"] = label_table[["Label", "Start", "Duration","End","wav_path","label_path"]]
                        
                        
                        # column_names = ["Label","Start","Duration","End"]
                        # column_names.extend(list(testing_label_dict.keys()))
                        # for file in label_list :
                        #     df = pd.read_csv(file, delimiter=';') 
                        #     # df = df.drop_duplicates(keep=False)
                        #     df = df[column_names]
                        #     df.to_csv(file, header=True, index=None, sep=';', mode = 'w')
                        
                        
                        no_call = set(["noise", "beep", "synch"])
                        
                        evaluation = metrics.Evaluate(label_list = label_list, 
                                                      prediction_list = pred_list, 
                                                      noise_label = "noise", 
                                                      IoU_threshold = 0.2,
                                                      GT_proportion_cut = 0.01, 
                                                      no_call = no_call,
                                                      category_list = set(["cc", "sn", "mo", "agg", "ld", "soc", "al", "beep", "synch","oth", "noise"]),
                                                      headers = set(['Label', 'Duration', 'Start', 'End']),
                                                      nonfoc_tags =["NONFOC", "nf", "*"],
                                                      start_labels = ['START','start'],
                                                      stop_labels = ['END', 'STOP', 'stop'],
                                                      skipon_labels = ['skipon', 'SKIPON'],
                                                      skipoff_labels = ['skipoff', 'SKIPOFF']
                                                      ) # 0.99 is 0.5
                        
                        
                        output, skipped_calls = evaluation.main()   
                        
                        
                        
                        
                        array = output["Confusion_Matrix"]  
                        df_cm = pd.DataFrame(array) #, range(6), range(6))    
                        
                        # get rid of the weird indentations and make rows and columns as names
                        '''
                        new_col = df_cm.iloc[0] # grab the first row for the header
                        df_cm = df_cm[1:] # take the data less the header row
                        df_cm.columns = new_col # set the header row as the df header    
                        new_row = df_cm['']
                        df_cm = df_cm.drop('', 1)
                        df_cm.index = new_row
                        '''
                        df_cm.index.name= None
                        df_cm.columns.name= None
                        
                        # # replace FP and FN with noise
                        df_cm['noise'] = df_cm['FN'] 
                        #df_cm.loc['noise']=df_cm.loc['FP']
                        
                        # remove FP and FN
                        df_cm = df_cm.drop("FN", axis=1)
                        #df_cm = df_cm.drop("FP", axis=0)
                        
                        df_cm = df_cm.apply(pd.to_numeric)
                        
                        # Raw confusion matrix
                        df_cm = df_cm[list(testing_label_dict.keys())]
                        df_cm = df_cm.reindex(list(testing_label_dict.keys()))     
                        
                        
                        
                        #########################################
                        # CALCULATE
                        
                        # Recall confusion matrix
                        df_recall = df_cm.div(df_cm.sum(axis=1), axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
                        
                        # Proportion of calls for confusion matrix
                        call_len = list()
                        for i in call_len_dict.keys():
                            call_len.append(call_len_dict[i])
                        # add noise at the end
                        call_len[-1] = df_cm.sum(axis=1)[-1]
                        
                        #proportion of calls
                        df_prop = df_cm.div(call_len, axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
                        
                        #########################################
                        # PLOT

                        
                        #multi figure parameters
                        fig,((ax1,ax2,ax3)) = plt.subplots(1,3, figsize=(20,5))
                        fig.suptitle("type: "+str(low_thr_1) + " - " + str(high_thr_1)+" / call: "+str(low_thr_2) + " - " + str(high_thr_2))
                        
                        # plot raw
                        sn.set(font_scale=1.1) # for label size
                        sn.heatmap((df_cm+1), annot=df_cm, fmt='g',norm = LogNorm(), annot_kws={"size": 10}, ax= ax1) # font size
                        ax1.set_title("Raw")              
                        
                        # plot recall
                        sn.set(font_scale=1.1) # for label size
                        sn.heatmap((df_recall), annot=True, fmt='g', annot_kws={"size": 10}, ax= ax2) # font size
                        ax2.set_title("Recall" )
                        
                        # plot proportion of calls
                        sn.set(font_scale=1.1) # for label size
                        sn.heatmap((df_prop), annot=True, fmt='g', annot_kws={"size": 10}, ax= ax3) # font size
                        ax3.set_title("Call Prop")
                        
                        # Save 3 panels
                        #plt.savefig(os.path.join(save_metrics_path, eval_analysis, "Confusion_mat_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
                        plt.show()
                        # plt.clf()
