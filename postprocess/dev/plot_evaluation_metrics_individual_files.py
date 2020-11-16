#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 19:03:49 2020

@author: kiran
"""



root_paths = ["/home/kiran/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_NoOther/new_evaluation_innerloop",
              "/home/kiran/Dropbox/CCAS_big_data/ML_data/NoiseAugmented_ProportionallyWeighted_NoOther/new_evaluation_innerloop",
              "/home/kiran/Dropbox/CCAS_big_data/ML_data/new_run_sep_2020/new_evaluation_innerloop"]

i = 1
# load the model
# RNN_model = keras.models.load_model(models_paths[i])    

# find the testing files for that model
with open( os.path.join(root_paths[i], "testing_files_used.txt")) as f:
    content = f.readlines()    
testing_filenames = [x.strip() for x in content] 
 
skipped_files = []
# for every test files for that model


with open( os.path.join(root_paths[i], "skipped_testing_files.txt")) as f:
    content = f.readlines()    
skipped_files = [x.strip() for x in content] 



pred_tables = glob.glob(root_paths[i]+"/pred_table"+ "/*PRED_TABLE*.txt")
for file in pred_tables:
    # file= pred_tables[80]
    df = pd.read_csv(file, delimiter=';') 
    # df = df.drop_duplicates(keep=False)
    df = df.loc[df['Label'] != 'Label']        
    new_filename = root_paths[i]+"/pred_table/edited/"+ os.path.basename(file)
    df.to_csv(new_filename, header=True, index=None, sep=';', mode = 'w')



##############################################################################################

#    EVALUATE
#
##############################################################################################

#########################################################################
##  Create overall thresholds
#########################################################################
overlap = 0.5
normalise = False
# skipped = [os.path.split(path)[1] for path in skipped_files]
file_ID_list =[file_ID for file_ID in testing_filenames if file_ID not in skipped_files]  #["HM_HRT_R09_AUDIO_file_5_(2017_08_24-06_44_59)_ASWMUX221110"]
# label_list =  [os.path.join(root_paths[i], "label_table", file_ID + "_LABEL_TABLE.txt" ) for file_ID in file_ID_list]
for file_ID in file_ID_list:
    for low_thr in [0.2]:
        for high_thr in [0.7]: 
            label_list =  [os.path.join(root_paths[i], "label_table", file + "_LABEL_TABLE.txt" ) for file in [file_ID]]
            low_thr = round(low_thr,2)                               
            high_thr = round(high_thr,2) 
            
            predictions_list = [os.path.join(root_paths[i], "pred_table", "edited", file + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) + ".txt" ) for file in [file_ID] ]
            evaluation = metrics.Evaluate(label_list, predictions_list, overlap, 5) # 0.99 is 0.5
            Prec, Rec, cat_frag, time_frag, cf, gt_indices, pred_indices, match, offset = evaluation.main()
            
            # specify file names
            precision_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Precision_2.csv'
            recall_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Recall_2.csv'
            cat_frag_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Category_fragmentation_2.csv'
            time_frag_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Time_fragmentation_2.csv'
            confusion_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Confusion_matrix_2.csv'
            gt_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + "_Label_indices_2.csv"
            pred_filename =file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + "_Prection_indices_2.csv"
            match_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + "_Matching_table_2.txt"
            timediff_filename = file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + "_Time_difference_2.txt"    
            
            # save files
            Prec.to_csv( os.path.join(root_paths[i], "metrics_old", precision_filename))
            Rec.to_csv( os.path.join(root_paths[i], "metrics_old", recall_filename))
            cat_frag.to_csv( os.path.join(root_paths[i], "metrics_old", cat_frag_filename))
            time_frag.to_csv(os.path.join(root_paths[i], "metrics_old", time_frag_filename))
            cf.to_csv(os.path.join(root_paths[i], "metrics_old", confusion_filename))
            gt_indices.to_csv(os.path.join(root_paths[i], "metrics_old", gt_filename ))
            pred_indices.to_csv(os.path.join(root_paths[i], "metrics_old", pred_filename ))                  
            with open(os.path.join(root_paths[i], "metrics_old", match_filename), "wb") as fp:   #Picklin
                      pickle.dump(match, fp)
            with open(os.path.join(root_paths[i], "metrics_old", timediff_filename), "wb") as fp:   #Pickling
                pickle.dump(offset, fp)    


    #########################################################################
    # plot overall confusion matrix
    #########################################################################
    
    for low_thr in [0.2]:
        for high_thr in [0.7]: 
            
            low_thr = round(low_thr,2)                               
            high_thr = round(high_thr,2) 
            confusion_filename = os.path.join(root_paths[i], "metrics_old", file_ID + "_PRED_TABLE_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_Confusion_matrix_2.csv')
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
            
            # # # replace FP and FN with noise
            # df_cm['noise'] = df_cm['FN'] 
            # df_cm.loc['noise']=df_cm.loc['FP']
            
            # # remove FP and FN
            # df_cm = df_cm.drop("FN", axis=1)
            # df_cm = df_cm.drop("FP", axis=0)
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
            ax.set_title(str(file_ID) + "-" + str(low_thr) + "-" + str(high_thr) )
            plt.savefig(os.path.join(root_paths[i], "metrics_old", file_ID + "Confusion_mat_thr_" + str(low_thr) + "-" + str(high_thr) +'_overlap_' + str(overlap) + '_2.png'))
            plt.show()