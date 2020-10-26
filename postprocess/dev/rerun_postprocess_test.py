#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:59:33 2020

@author: kiran
"""
import os
os.chdir("/home/baboon/Documents/github/CCAS_ML")



import keras
import os
import glob
from itertools import compress
import ntpath
import numpy as np
import pandas as pd
import warnings
import librosa
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

import preprocess.preprocess_functions as pre
import postprocess.merge_predictions_functions as ppm
import postprocess.evaluation_metrics_functions as metrics

# Find the input data
#-----------------------------------------------------------------
label_dirs = ["/home/baboon/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2017/labels_CSV", #2017 meerkats
            "/home/baboon/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2019/labels_CSV"]
audio_dirs= ["/home/baboon/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2017",
             "/home/baboon/Dropbox/CCAS_big_data/meerkat_data/meerkat_data_2019"]

#------------------
# label munging parameters i.e. reading in audition or raven files
sep='\t'
engine = None
start_column = "Start"
duration_column = "Duration"
label_column = "Name"
convert_to_seconds = True
label_for_other = "oth"
label_for_noise = "noise"
label_for_startstop = ['start', 'stop', 'skip', 'end']

normalise = True
#------------------
# call dictionary - 
# this is a dictionary containing as keys the category you want your ML algo to output
# and for each call category, how it is likely to be noted in the label column of the audition or raven file
# For example, Marker is usually for a close call.
# Note that these are regural expressions and are not case sensitive
call_types = {
    'cc' :["cc","Marker", "Marque"],
    'sn' :["sn","subm", "short","^s$", "s "],
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL", "so "],
    'al' :["al","ALARM"],
    'beep':["beep"],
    'synch':["sync"],
    'oth':["oth","other","lc", "lost",
           "hyb","HYBRID","fu","sq","\+",
           "ukn","unknown",          
           "x",
           "\%","\*","\#","\?","\$"
           ],
    'noise':['start','stop','end','skip']
    }

#------------------


# find all label paths
EXT = "*.csv"
label_filepaths = []
for PATH in label_dirs:
     label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])


# find all audio paths (will be longer than label path as not everything is labelled)
audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
     audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])
# get rid of the focal follows (going around behind the meerkat with a microphone)
audio_filepaths = list(compress(audio_filepaths, ["SOUNDFOC" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["PROCESSED" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["LABEL" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["label" not in filei for filei in audio_filepaths]))
audio_filepaths = list(compress(audio_filepaths, ["_SS" not in filei for filei in audio_filepaths]))


# Find the names of the recordings
audio_filenames = [os.path.splitext(ntpath.basename(wavi))[0] for wavi in audio_filepaths]
label_filenames = []
for filepathi in label_filepaths:
    for audio_nami in audio_filenames:
        if audio_nami in filepathi:
            label_filenames.append(audio_nami)





# had to create the testing filenames for the old run as it was not created
# testing_filenames = [filei.split("_LABEL_")[0] for filei in os.listdir("/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/label_table")]
# with open(os.path.join("/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/test_data/", "testing_files_used.txt"), "w") as f:
#     for s in testing_filenames:
#         f.write(str(s) +"\n") 

#-----------------------------------------------------

root_paths = ["/home/baboon/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_NoOther/new_evaluation",
              "/home/baboon/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_ProportionallyWeighted_NoOther/new_evaluation",
              "/home/baboon/Dropbox/CCAS_big_data/ML_data/new_run_sep_2020/new_evaluation"]

models_paths = ["/home/baboon/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_NoOther/new_evaluation/NoiseAugmented_NoOther_2020-10-06_22:35:04.511239/savedmodel.h5",
                "/home/baboon/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_ProportionallyWeighted_NoOther/new_evaluation/NoiseAugmented_ProportionallyWeighted_NoOther_2020-10-14_03:12:32.817594/savedmodel.h5",    
                "/home/baboon/Dropbox/CCAS_big_data/ML_data/new_run_sep_2020/new_evaluation/model_2020-09-15_18:57:17.170622/savedmodel.h5"]




# for every model run
for i in range(0,len(models_paths)):    
    # load the model
    RNN_model = keras.models.load_model(models_paths[i])    
    
    # find the testing files for that model
    with open( os.path.join(root_paths[i], "testing_files_used.txt")) as f:
        content = f.readlines()    
    testing_filenames = [x.strip() for x in content] 
     
    skipped_files = []
    # for every test files for that model
    for file_ID in testing_filenames:
        
        # find the matching audio for the label data
        audio_path = [s for s in audio_filepaths if file_ID in s][0]
        
        # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
        label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
        
        print("*****************************************************************")   
        print("*****************************************************************") 
        print ("File being processed : " + label_path)
        
        # create a standardised table which contains all the labels of that file - also can be used for validation
        label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, 
                                       convert_to_seconds, label_for_other, label_for_noise, engine, True)
        # replace duration of beeps with 0.04 seconds - meerkat particularity
        label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
        label_table.loc[label_table["beep"] == True, "End"] += 0.04
        
        # find the start and stop  of the labelling periods (also using skipon/skipoff)
        loop_table = label_table.loc[label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False), ["Label","Start"]]
        loop_times = list(loop_table["Start"])
        
        # Make sure that the file contains the right number of start and stops, otherwise go to the next file
        if len(loop_times)%2 != 0:
            print("!!!!!!!!!!!!!!!!")
            warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
            skipped_files.append(file_ID)
            # break
            continue 
        if len(loop_times) == 0:
            print("!!!!!!!!!!!!!!!!")
            warnings.warn("There is a missing start or stop in this file and it has been skipped: " + label_path)
            skipped_files.append(file_ID)
            # break
            continue 
        
        # save the label_table
        save_label_table_filename = file_ID + "_LABEL_TABLE.txt"
        
        # If the file hasn't already been processed save it
        if not os.path.isfile(os.path.join(root_paths[i], "label_table", save_label_table_filename)):
            label_table.to_csv(os.path.join(root_paths[i], "label_table", save_label_table_filename), 
                               header=True, index=None, sep=';')
        
        
        # load the audio data
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        
        # # Reshaping the Audio file (mono) to deal with all wav files similarly
        # if y.ndim == 1:
        #     y = y.reshape(1, -1)
        
        # # Implement this for acc data
        # for ch in range(y.shape[0]):
        # ch=0
        # y_sub = y[:,ch]
        y_sub = y
        
        # probabilities = []
        # for low_thr in [0.2]:
        # loop through every labelling start based on skipon/off within this loop_table
        for loopi in range(0, int(len(loop_times)), 2):
            # loopi = 0
            fromi =  loop_times[loopi]
            #toi = fromi + 5
            toi = loop_times[int(loopi + 1)] # define the end of the labelling periods
            
            # if the file exists, load it
            if os.path.exists(os.path.join(root_paths[i],"predictions", file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy')):
                pred_list = np.load( os.path.join(root_paths[i],"predictions", file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'))
            # if not, generate it
            else:
            
                pred_list = []
        
                for spectro_slide in np.arange(fromi, toi, slide):                    
                    
                    # spectro_slide = fromi
                    start = round(spectro_slide,3)
                    stop = round(spectro_slide + spec_window_size, 3)
                    
                    # start = round(start + slide, 3)
                    # stop = round(spectro_slide + spec_window_size, 3)
                    # ignore cases where the window is larger than what is labelled (e.g. at the end)
                    if stop <= toi:
                        
                        # # Generate the relevant spectrogram name
                        # save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                        # save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                        # save_pred_filename = file_ID + "_PRED_" + str(start) + "s-" + str(stop) + "s_" #+ category + ".npy"
                        
                        spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                                               n_mels = n_mels, window='hann', 
                                                               fft_win= fft_win, fft_hop = fft_hop, normalise = True)
                        
                        label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                                               stop, label_for_noise)
                        
                        # Load the spectrogram
                        spec = spectro.T
                        spec = spec[np.newaxis, ..., np.newaxis]  
                        
                        # generate the prediction
                        pred = RNN_model.predict(spec)
                        
                        # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
                        # file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
                        # if len(file_label) > 1 and 'noise' in file_label:
                        #     file_label.remove('noise')
                        # category = '_'.join(file_label)
                        
                        # save_spec_filename = save_spec_filename + category + ".npy"
                        # save_mat_filename = save_mat_filename + category + ".npy"
                        # save_pred_filename = save_pred_filename + category + ".npy"
                        
                        # add this prediction to the stack that will be used to generate the predictions table
                        pred_list.append(np.squeeze(pred))
                        
                # save the prediction list  
                np.save( os.path.join(root_paths[i],"predictions" , file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.npy'), pred_list)
                with open(os.path.join(root_paths[i], "predictions" ,file_ID + '_PRED_STACK_' + str(fromi) + '-' + str(toi) + '.txt'), "w") as f:
                    for row in pred_list:
                        f.write(str(row) +"\n")
                    
            for low_thr in [0.2]:
                for high_thr in [0.5,0.7,0.9]: 
                    
                    low_thr = round(low_thr,2)                               
                    high_thr = round(high_thr,2)
    
                    save_pred_table_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt"
                    
                    #----------------------------------------------------------------------------
                    # Compile the predictions for each on/off labelling chunk
                    detections = ppm.merge_p(probabilities = pred_list, 
                                             labels=list(call_types.keys()),
                                             starttime = 0, 
                                             frameadv_s = fft_hop, 
                                             specadv_s = slide,
                                             low_thr=low_thr, 
                                             high_thr=high_thr, 
                                             debug=1)
                    
                    if len(detections) == 0:  
                        detections = pd.DataFrame(columns = ['category', 'start', 'end', 'scores'])
                    
                    pred_table = pd.DataFrame() 
                    
                    #convert these detections to a predictions table                
                    table = pd.DataFrame(detections)
                    table["Label"] = table["category"]
                    table["Start"] = round(table["start"]*fft_hop + fromi, 3)#table["start"].apply(Decimal)*Decimal(fft_hop) + Decimal(fromi)
                    table["Duration"] = round( (table["end"]-table["start"])*fft_hop, 3)#(table["end"].apply(Decimal)-table["start"].apply(Decimal))*Decimal(fft_hop)
                    table["End"] = round(table["end"]*fft_hop + fromi, 3)#table["Start"].apply(Decimal) + table["Duration"].apply(Decimal)
                    
                    #keep only the useful columns
                    # table = table[["Label","Start","Duration", "End"]]            
                    table = table[["Label","Start","Duration", "End", "scores"]]  
                    
                    # Add a row which stores the start of the labelling period
                    row_start = pd.DataFrame()
                    row_start.loc[0,'Label'] = list(loop_table["Label"])[loopi]
                    row_start.loc[0,'Start'] = fromi
                    row_start.loc[0,'Duration'] = 0
                    row_start.loc[0,'End'] = fromi 
                    row_start.loc[0,'scores'] = None
                    
                    # Add a row which stores the end of the labelling period
                    row_stop = pd.DataFrame()
                    row_stop.loc[0,'Label'] = list(loop_table["Label"])[int(loopi + 1)]
                    row_stop.loc[0,'Start'] = toi
                    row_stop.loc[0,'Duration'] = 0
                    row_stop.loc[0,'End'] = toi 
                    row_start.loc[0,'scores'] = None
                    
                    # put these rows to the label table
                    table = pd.concat([row_start, table, row_stop]) 
                    
                    # add the true false columns based on the call types dictionary
                    for true_label in call_types:
                        table[true_label] = False
                        for old_label in call_types[true_label]:
                            table.loc[table["Label"].str.contains(old_label, regex=True, case = False), true_label] = True
                    
                    # add this table to the overall predictions table for that collar
                    pred_table = pd.concat([pred_table, table ])
                    
                    
                    # for each on/off labelling chunk, we can save the prediction and append it to the previous chunk
                    pred_table.to_csv(os.path.join(root_paths[i], "pred_table", save_pred_table_filename), 
                                      header=True, index=None, sep=';', mode = 'a')
                    
                    
                    

            
    # save the files that were skipped
    print(skipped_files)
    
    # save a copy of the training and testing diles
    with open(os.path.join(root_paths[i], "skipped_testing_files.txt"), "w") as f:
        for s in skipped_files:
            f.write(str(s) +"\n")
           
    ##############################################################################################
    #Loop through tables and remove duplicates of rows (bevause files are created through appending)
    pred_tables = glob.glob(root_paths[i]+ "/*PRED_TABLE*.txt")
    for file in pred_tables:
        df = pd.read_csv(file, delimiter=';') 
        # df = df.drop_duplicates(keep=False)
        df = df[df['Label'] != 'Label']
        df.to_csv(file, header=True, index=None, sep=';', mode = 'w')
    
    
    ##############################################################################################
    #
    #    EVALUATE
    #
    ##############################################################################################
    
    #########################################################################
    ##  Create overall thresholds
    #########################################################################
    
    # skipped = [os.path.split(path)[1] for path in skipped_files]
    file_ID_list = [file_ID for file_ID in testing_filenames if file_ID not in skipped_files]
    label_list =  [os.path.join(root_paths[i], "label_table", file_ID + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]
    for low_thr in [0.2]:#[0.1,0.3]:
        for high_thr in [0.5,0.7,0.9]: 
            
            low_thr = round(low_thr,2)                               
            high_thr = round(high_thr,2) 
            
            pred_list = [os.path.join(root_paths[i], "pred_table", file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt" ) for file_ID in file_ID_list ]
            evaluation = metrics.Evaluate(label_list, pred_list, 0.5, 5) # 0.99 is 0.5
            Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
            
            # specify file names
            precision_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Precision.csv'
            recall_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Recall.csv'
            cat_frag_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Category_fragmentation.csv'
            time_frag_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Time_fragmentation.csv'
            confusion_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv'
            gt_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Label_indices.csv"
            pred_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Prection_indices.csv"
            match_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Matching_table.txt"
            timediff_filename = "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + "_Time_difference.txt"    
            
            # save files
            Prec.to_csv( os.path.join(root_paths[i], "metrics", precision_filename))
            Rec.to_csv( os.path.join(root_paths[i], "metrics", recall_filename))
            cat_frag.to_csv( os.path.join(root_paths[i], "metrics", cat_frag_filename))
            time_frag.to_csv(os.path.join(root_paths[i], "metrics", time_frag_filename))
            cf.to_csv(os.path.join(root_paths[i], "metrics", confusion_filename))
            gt_indices.to_csv(os.path.join(root_paths[i], "metrics", gt_filename ))
            pred_indices.to_csv(os.path.join(root_paths[i], "metrics", pred_filename ))                  
            with open(os.path.join(root_paths[i], "metrics", match_filename), "wb") as fp:   #Picklin
                      pickle.dump(match, fp)
            with open(os.path.join(root_paths[i], "metrics", timediff_filename), "wb") as fp:   #Pickling
                pickle.dump(offset, fp)    
    
    
    #########################################################################
    # plot overall confusion matrix
    #########################################################################
    
    for low_thr in [0.2]:
        for high_thr in [0.5,0.7,0.9]: 
            
            low_thr = round(low_thr,2)                               
            high_thr = round(high_thr,2) 
            confusion_filename = os.path.join(root_paths[i], "metrics", "Overall_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + '_Confusion_matrix.csv')
            with open(confusion_filename, newline='') as csvfile:
                array = list(csv.reader(csvfile))
        
            df_cm = pd.DataFrame(array)#, range(6), range(6))    
            
            #get rid of the weird indentations and make rows and columns as names
            new_col = df_cm.iloc[0] #grab the first row for the header
            df_cm = df_cm[1:] #take the data less the header row
            df_cm.columns = new_col #set the header row as the df header    
            new_row = df_cm['']
            df_cm = df_cm.drop('', 1)
            df_cm.index = new_row
            df_cm.index.name= None
            df_cm.columns.name= None
            
            # # replace FP and FN with noise
            df_cm['noise'] = df_cm['FN'] 
            df_cm.loc['noise']=df_cm.loc['FP']
            
            # remove FP and FN
            df_cm = df_cm.drop("FN", axis=1)
            df_cm = df_cm.drop("FP", axis=0)
            ####
            
            
            df_cm = df_cm.apply(pd.to_numeric)
            # #move last negatives to end
            # col_name = "FN"
            # last_col = df_cm.pop(col_name)
            # df_cm.insert(df_cm.shape[1], col_name, last_col)
            
            # # remove noi        for low_thr in [0.1,0.3]:
                # for high_thr in [0.5,0.7,0.8,0.9,0.95]: 
            
            #normalise the confusion matrix
            if normalise == True:
                # divide_by = df_cm.sum(axis=1)
                # divide_by.index = new_header
                # new_row = df_cm.index 
                # new_col = df_cm.columns
                df_cm = df_cm.div(df_cm.sum(axis=1), axis=0).round(2)#pd.DataFrame(df_cm.values / df_cm.sum(axis=1).values).round(2)
                # df_cm.index = new_row
                # df_cm.columns = new_col
            
            # plt.figure(figsize=(10,7))
            ax = plt.axes()
            sn.set(font_scale=1.1) # for label size
            sn.heatmap((df_cm), annot=True, annot_kws={"size": 10}, ax= ax) # font size
            ax.set_title(str(low_thr) + "-" + str(high_thr) )
            plt.savefig(os.path.join(root_paths[i], "metrics", "Confusion_mat_thr_" + str(low_thr) + "-" + str(high_thr) + '.png'))
            plt.show()
    
    
    
    

