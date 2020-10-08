import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas import DataFrame
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from evaluation.fragments import get_fragments, plot_fragments

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)


class Evaluate:
    def __init__(self, model_to_evaluate, dataset_dir, species, is_three_axis, batch_size, threshold):
        self.model = load_model(model_to_evaluate)
        self.dataset_dir = dataset_dir
        self.is_three_axis = is_three_axis
        self.x_test_aud = np.load(os.path.join(dataset_dir, "dataset_aud/x_test.npy"))
        self.x_test_acc_ch0 = np.load(os.path.join(dataset_dir, "dataset_acc_ch_0/x_test.npy"))
        self.x_test_acc_ch1 = np.load(os.path.join(dataset_dir, "dataset_acc_ch_1/x_test.npy"))
        self.x_test_acc_ch2 = np.load(os.path.join(dataset_dir, "dataset_acc_ch_2/x_test.npy"))

        self.y_test_aud = np.load(os.path.join(dataset_dir, 'dataset_aud/y_test_aud.npy'))
        self.y_test_foc = np.load(os.path.join(dataset_dir, 'dataset_aud/y_test_foc.npy'))
        self.test_files = np.load(os.path.join(dataset_dir, 'dataset_aud/test_files.npy'))
        self.batch_size = batch_size
        self.species = species
        self.threshold = threshold
        
        
    def get_call_ranges(self,y):
        call_ranges = []
        for frame in y:
            indices = self.get_window_call_range(frame)
            call_ranges.append(indices)
        return np.asarray(call_ranges)    
        
    def get_window_call_range(self,frame):
        ''' For every window, determine the start and end timesteps of continuous calls in the label'''
        indices = []
        arr = frame[:, :8].T
        for i in range(len(arr)):
            call_indices = np.where(arr[i] != 0)
            sequences = np.split(call_indices[0], np.where(np.diff(call_indices[0]) > 1)[0] + 1)
            timesteps = []
            for s in sequences:
                if len(s) > 0: timesteps.append([s[0], s[len(s) - 1]])
            indices.append(timesteps)
        return indices   
    
    def precision(self, TP,FP):
        Prec = np.zeros(np.size(TP))
        for i in range(np.size(TP)):
            Prec[i] = TP[i] / (TP[i] + FP[i])
        return Prec
    
    def recall(self, TP,FN):
        Rec = np.zeros(np.size(TP))
        for i in range(np.size(TP)):
            Rec[i] = TP[i] / (TP[i] + FN[i])
        return Rec
    
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
        match = np.zeros([np.size(gt_indices,1)+1, np.size(gt_indices,1)+1,np.size(gt_indices,0)])
        # The first loop compares the list of calls to all detected calls for every recording.
        for call in range(np.size(gt_indices,1)):
            for idx in range(np.size(gt_indices,0)):
                for call_nb in range(len(gt_indices[idx,call])):
                    appel = gt_indices[idx,call][call_nb]
                    detected = False
                    call_start = appel[0]
                    call_end = appel[1]
                    # The call is compared to all detected calls in the recording
                    for pred in range(np.size(pred_indices,1)):
                        for pred_nb in range(len(pred_indices[idx,pred])):
                            detected_start = pred_indices[idx,pred][pred_nb][0]
                            detected_end = pred_indices[idx,pred][pred_nb][1]
                            # Are the calls simultaneous? 
                            if ((call_start <= detected_end and detected_end <= call_end)
                                  or (call_start <= detected_start and detected_start <= call_end)
                                  or (detected_start <= call_start and call_end <= detected_end)):
                                intersection = min(call_end, detected_end) - max(call_start, detected_start)
                                union = max(call_end, detected_end) - min(call_start, detected_start)
                                if(intersection / union > self.threshold):
                                    detected = True
                                    if(call == pred):
                                        match[call,call,idx].append([pred_nb,call_nb])
                                    else:
                                        fp = 1
                                        other_call = 0
                                        # The following loop ensures that the detected call is indeed a false positive,
                                        # rather than the detection of a different actual call at the same time.
                                        # It compares with the calls of that type whether one was emitted at that time.
                                        while(fp == 1 and other_call < len(gt_indices[idx,pred])):
                                            other_call_start = gt_indices[idx,pred][other_call][0]
                                            other_call_end = gt_indices[idx,pred][other_call][1]
                                            if ((other_call_start <= detected_end and detected_end <= other_call_end) 
                                                or (other_call_start <= detected_start and detected_start <= other_call_end) 
                                                or (detected_start <= other_call_start and other_call_end <= detected_end)):
                                                intersection = min(other_call_end, detected_end) - max(other_call_start, detected_start)
                                                union = max(other_call_end, detected_end) - min(other_call_start, detected_start)
                                                if(intersection / union > self.threshold):
                                                    fp = 0
                                            other_call = other_call + 1
                                        if(fp == 1):
                                            match[pred,call,idx].append([pred_nb,call_nb])
                    if(not detected):
                        match[len(match)-1,call,idx].append([np.nan,call_nb])
            
        
        # The second loop compares the detected calls to the list of actual calls, in order to find false positives
        for pred in range(np.size(pred_indices,1)):
            for idx in range(np.size(pred_indices,0)):
                for pred_nb in range(len(pred_indices[idx,pred])):
                    fp = True                    
                    predit = pred_indices[idx,pred][pred_nb]
                    detected_start = predit[0]
                    detected_end = predit[1]    
                    call = 0
                    while(fp and call < np.size(gt_indices,1)):
                        call_nb = 0
                        while(fp and call_nb < len(gt_indices[idx,call])):
                            call_start = gt_indices[idx,call][call_nb][0]
                            call_end = gt_indices[idx,call][call_nb][1]
                            if ((call_start <= detected_end and detected_end <= call_end)
                                  or (call_start <= detected_start and detected_start <= call_end)
                                  or (detected_start <= call_start and call_end <= detected_end)):
                                intersection = min(call_end, detected_end) - max(call_start, detected_start)
                                union = max(call_end, detected_end) - min(call_start, detected_start)
                                if(intersection / union > self.threshold):
                                    fp = False
                            call_nb = call_nb + 1
                        call = call + 1
                    if(fp):
                        match[pred,len(match[pred])-1,idx].append([pred,np.nan])
        return match
    
    def get_confusion_matrix(self, match):
        '''Generates the confusion matrix corresponding to the list of calls gt_indices 
        and the list of predicted calls pred_indices.
        If there are N types of calls, the confusion matrix's size is (N+1)*(N+1);
        the last line and column correspond to the false negatives and false positives respectively,
        in the sense that the call wasn't detected at all (and not merely detected as a different type of call),
        or that the detection doesn't match an actual call.
        '''
        cf = np.zeros([np.size(match,1), np.size(match,1)])
        # The first loop compares the list of calls to all detected calls for every recording.
        for call in range(np.size(match,1)):
            for idx in range(np.size(gt_indices,0)):
                for call_nb in range(len(gt_indices[idx,call])):
                    appel = gt_indices[idx,call][call_nb]
                    call_start = appel[0]
                    call_end = appel[1]
                    # The call is compared to all detected calls in the recording
                    for pred in range(np.size(pred_indices,1)):
                        for pred_nb in range(len(pred_indices[idx,pred])):
                            detected_start = pred_indices[idx,pred][pred_nb][0]
                            detected_end = pred_indices[idx,pred][pred_nb][1]
                            # Are the calls simultaneous? 
                            if ((call_start <= detected_end and detected_end <= call_end)
                                  or (call_start <= detected_start and detected_start <= call_end)
                                  or (detected_start <= call_start and call_end <= detected_end)):
                                intersection = min(call_end, detected_end) - max(call_start, detected_start)
                                union = max(call_end, detected_end) - min(call_start, detected_start)
                                if(intersection / union > self.threshold):
                                    if(call == pred):
                                        cf[call,call] = cf[call,call] + 1
                                    else:
                                        fp = 1
                                        other_call = 0
                                        # The following loop ensures that the detected call is indeed a false positive,
                                        # rather than the detection of a different actual call at the same time.
                                        # It compares with the calls of that type whether one was emitted at that time.
                                        while(fp == 1 and other_call < len(gt_indices[idx,pred])):
                                            other_call_start = gt_indices[idx,pred][other_call][0]
                                            other_call_end = gt_indices[idx,pred][other_call][1]
                                            if ((other_call_start <= detected_end and detected_end <= other_call_end) 
                                                or (other_call_start <= detected_start and detected_start <= other_call_end) 
                                                or (detected_start <= other_call_start and other_call_end <= detected_end)):
                                                intersection = min(other_call_end, detected_end) - max(other_call_start, detected_start)
                                                union = max(other_call_end, detected_end) - min(other_call_start, detected_start)
                                                if(intersection / union > self.threshold):
                                                    fp = 0
                                            other_call = other_call + 1
                                        cf[np.size(gt_indices,1),call] = cf[np.size(gt_indices,1),call] + fp
            
        
        # The second loop compares the detected calls to the list of actual calls, in order to find false positives
        for pred in range(np.size(pred_indices,1)):
            for idx in range(np.size(pred_indices,0)):
                for pred_nb in range(len(pred_indices[idx,pred])):
                    fn = 1                    
                    predit = pred_indices[idx,pred][pred_nb]
                    detected_start = predit[0]
                    detected_end = predit[1]    
                    call = 0
                    while(fn == 1 and call < np.size(gt_indices,1)):
                        call_nb = 0
                        while(fn == 1 and call_nb < len(gt_indices[idx,call])):
                            call_start = gt_indices[idx,call][call_nb][0]
                            call_end = gt_indices[idx,call][call_nb][1]
                            if ((call_start <= detected_end and detected_end <= call_end)
                                  or (call_start <= detected_start and detected_start <= call_end)
                                  or (detected_start <= call_start and call_end <= detected_end)):
                                intersection = min(call_end, detected_end) - max(call_start, detected_start)
                                union = max(call_end, detected_end) - min(call_start, detected_start)
                                if(intersection / union > self.threshold):
                                    fn = 0
                            call_nb = call_nb + 1
                        call = call + 1
                    cf[pred,np.size(gt_indices,1)] = cf[pred,np.size(gt_indices,1)] + fn
        
        return(cf)



    def main(self):

        if self.is_three_axis:
            preds_w_threshold = self.model.predict([self.x_test_aud, self.x_test_acc_ch0, self.x_test_acc_ch1, self.x_test_acc_ch2], self.batch_size)
        else:
            preds_w_threshold = self.model.predict([self.x_test_aud, self.x_test_acc_ch2], self.batch_size)
        if(self.species=="hyenas"):
            class_names = [['GIG', 'SQL', 'GRL', 'GRN', 'SQT', 'MOO', 'RUM', 'WHP'], ['NON-FOC', 'NOTDEF', 'FOC']]
            type = ['Call Type', 'Focal Type']
        y_test = [self.y_test_aud, self.y_test_foc]
        
        # Considering noise label as the threshold for call label predictions
        for idx, frame in enumerate(preds_w_threshold[0]):
            for row in frame:
                row[:8][row[:8] < row[8]] = 0  # making every sound with less than noise amplitude equal to 0
        preds_w_threshold[0][preds_w_threshold[0] > 0] = 1

        # Considering 0.5 as the threshold for the Focal Type predictions
        preds_w_threshold[1][preds_w_threshold[1] > 0.5] = 1
        preds_w_threshold[1][preds_w_threshold[1] <= 0.5] = 0
            
        gt_indices = self.get_call_ranges(self.y_test_aud)
        pred_indices = self.get_call_ranges(preds_w_threshold[0])      
        
        match = self.match_prediction_to_labels(gt_indices, pred_indices)
        cf = self.get_confusion_matrix(self, match)
        
        TP = np.zeros(np.size(gt_indices,1))
        FN = np.zeros(np.size(gt_indices,1)+1)
        FP = np.zeros(np.size(gt_indices,1)+1)
        for call in range(np.size(gt_indices,1)):
            TP[call] = cf[call,call]
            FN[call] = np.sum(cf[:,call])
            FP[call] = np.sum(cf[call,:])
        FN[np.size(gt_indices,1)] = sum(cf[np.size(gt_indices,1),:]) 
        FP[np.size(gt_indices,1)] = sum(cf[:,np.size(gt_indices,1)]) 
        print(cf)
        
        Prec = precision(self, TP,FP)
        Rec = recall(self, TP,FN)
                                    
                                    
                                
                                
                    
                    
                
            
            
            


if __name__=="__main__":
    evaluate = Evaluate("/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/saved_models/model_2019-12-21_21_02_27.534641_network_train/savedmodel.h5",
                        "/home/mathieu/Documents/Detecting-and-Classifying-Animal-Calls/rmishra/dataset","hyenas",
                        False, 64, 0.5)
    evaluate.main()
