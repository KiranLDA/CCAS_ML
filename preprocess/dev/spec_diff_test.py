#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:44:39 2020

@author: kiran
"""
import librosa

# Meerkat data location
audio_path = "/media/kiran/Kiran Meerkat/meerkat_detector/data/raw_data/AUDIO2/HM_LT_R07_20170821-20170825/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092.wav"


spec_window_size =  1#5.11 0.5#
slide = 0.75#2.56


y, sr = librosa.load(audio_path, sr=None, mono=False)

# Reshaping the Audio file (mono) to deal with all wav files similarly
if y.ndim == 1:
    y = y.reshape(1, -1)

# for ch in range(y.shape[0]):
ch=0
length = int(len(y[ch]) / sr)
remainder = length % spec_window_size#self.window_size


# for i in range(0, length - remainder - spec_window_size, slide):
i=3681#0#5334#3646#3697#3683#
start = i
stop = i + spec_window_size

### function start ###
# generate_mel_spectrogram
# spectro = get_spectrogram(begin_time, end_time, y[ch], sr)
y_sub = y[ch]


#---------------------------------------------------------------------------
# function for generating spectrogram

def generate_mel_spectrogram(y, sr, start, stop, n_mels = 128, window='hann', 
                             win_length = 0.005, hop_length = 0.0025):
    ''' Generate a mel frequency spectrogram
        
        The input will be an image of the sound which is generated using a spectrogram. This generates the spectrogram
        y:
            sound dataframe containing values from wav file 
        sr:
            sample rate in Hz
        start:
            start time be used to subset y and generate spectrogram (in seconds)
        stop:
            end timeto be used to subset y and generate spectrogram (in seconds)
        n_mels: 
            number of mel bands - suggested 64 or 128
        window: 
            spectrogram window generation type - default is "hanning"
        window_length: 
            window length (in seconds)
        hop_length: 
            hop between window starts (in seconds)
        
    '''
    # function dependencies
    import librosa
    import librosa.display
    import numpy as np
    # import os
    
    
    win_length  = int(win_length * sr) 
    hop_length = int(hop_length * sr) 
    
    start = int(sr * float(start))
    stop = int(sr * float(stop))
    
    # start= 0
    # stop = start + 512
    data_subset = np.asfortranarray(y[start:stop])
    # data_subset.shape
    
    s = librosa.feature.melspectrogram(y = data_subset ,
                                       sr = sr, 
                                       n_mels = n_mels , 
                                       fmax = sr/2, 
                                       hop_length = hop_length, 
                                       window = window, 
                                       win_length = win_length )
    spectro = librosa.power_to_db(s, ref=np.max)
    # print(spectro.shape)
    return spectro
    ### function end ###


# --------------------------------------------------------------

import preprocess.preprocess_functions as pre

# Test plot a spectrogram

win_length = 0.01 #0.03 #0.03#
hop_length= win_length/2#win_length/8 #0.002
n_mels = 30 #128
# win_length = 0.005#0.02
# hop_length= win_length/2#0.002


spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, n_mels = n_mels , 
                                window='hann', fft_win = win_length, fft_hop = hop_length, normalise = False)
print(spectro.shape)



### Plotting for simplicity
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
# spectro = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(spectro, x_axis='time',
                          y_axis='mel', sr=sr,
                          fmax=sr / 2)
plt.colorbar(format='%+2.0f dB')
plt.title('win=' + str(win_length) + ',hop=' + str(hop_length) )
plt.tight_layout()
plt.show()


#----------------------------------------
win_length = 0.03 #0.03#
hop_length= win_length/8#win_length/8 #0.002
n_mels = 128
# win_length = 0.005#0.02
# hop_length= win_length/2#0.002


spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, n_mels = n_mels , 
                                window='hann', fft_win = win_length, fft_hop = hop_length, normalise = False)
print(spectro.shape)



### Plotting for simplicity
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
# spectro = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(spectro, x_axis='time',
                          y_axis='mel', sr=sr,
                          fmax=sr / 2)
plt.colorbar(format='%+2.0f dB')
plt.title('win=' + str(win_length) + ',hop=' + str(hop_length) )
plt.tight_layout()
plt.show()



# z = np.median(spectro, axis=1)
                                                                                                                                                                                                                                                                                                                                                                                        # X = spectro
Y = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, n_mels = n_mels , 
                                window='hann', fft_win = win_length, fft_hop = hop_length, normalise = True)
# Y = spectro - spectro.mean(axis=0, keepdims=True)

# for col in range(spectro.shape[1]):
#     specdiff = spectro[:,col] - z[col]


plt.figure(figsize=(10, 7))
# plt.plot(z)
librosa.display.specshow(Y, x_axis='time',
                          y_axis='mel', sr=sr,
                          fmax=sr / 2)
plt.colorbar(format='%+2.0f dB')
plt.title('win=' + str(win_length) + ',hop=' + str(hop_length) )
plt.tight_layout()
plt.show()    



# label_path = "/media/kiran/Kiran Meerkat/meerkat_detector/ground_truth/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092_label.csv"
label_path = "/home/kiran/Documents/ML/labels_CSV/labels_CSV/HM_LT_R07_AUDIO_file_6_(2017_08_25-06_44_59)_ASWMUX221092_label.csv"

#------------------------------------------------------------------
# Time conversion functions
from decimal import *
def get_sec_long(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return float(int(h) * 3600 + int(m) * 60 + Decimal(s))
print(get_sec_long('1:23:45.127'))



from decimal import *
def get_sec_short(time_str):
    """Get Seconds from time."""
    m, s = time_str.split(':')
    return float(int(m) * 60 + Decimal(s))
print(get_sec_short('23:45.127'))





#------------------------------------------------------------------------


#-----------------------------------------------------------------------------
# call type labels
#-----------------------------------------------------------------------------

# call_types = {
#     'cc' :["cc","CC","Marker"],
#     'sn' :["sn","SN","subm", "short","SHORT"],
#     'mo' :["mo","MO","mov","MOV","move","MOVE"],
#     'agg':["ag","AG","agg","AGG","AGGRESS","chat","CHAT","GROWL","growl"],
#     'ld' :["ld","LD","lead","LEAD"],
#     'soc':["soc","SOC","social","SOCIAL"],
#     'al' :["al","AL","alarm","ALARM"],
#     'beep':["beep","BEEP"],
#     'synch':["synch", "SYNCH"],
#     'oth':["oth","other","OTH","OTHER","lc",
#            "hyb","HYB","hybrid","HYBRID","fu","sq","+",
#            "ukn","unknown","UKN","UNKNOWN",
#            "nf","nonfoc","NONFOC",
#            "x","X",
#            "%","*","#","?"]}
#     # 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
#     # 'ukn':["ukn","unknown","UKN","UNKNOWN"]
#     # 'nf' :["nf","nonfoc","NONFOC"],
#     # 'noise':["x","X"]
#     # 'overlap':"%"

call_types = {
    'cc' :["cc","Marker"],
    'sn' :["sn","subm", "short",],
    'mo' :["mo","MOV","MOVE"],
    'agg':["AG","AGG","AGGRESS","CHAT","GROWL"],
    'ld' :["ld","LD","lead","LEAD"],
    'soc':["soc","SOCIAL"],
    'al' :["al","ALARM"],
    'beep':["beep"],
    'synch':["synch"],
    'oth':["oth","other","lc",
           "hyb","HYBRID","fu","sq","+",
           "ukn","unknown",
           "nf","nonfoc",
           "x",
           "%","*","#","?"],
    'noise':['start','stop','end','skip']
    }
    # 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
    # 'ukn':["ukn","unknown","UKN","UNKNOWN"]
    # 'nf' :["nf","nonfoc","NONFOC"],
    # 'noise':["x","X"]
    # 'overlap':"%"

import pandas as pd

def create_meerkat_table(label_path, call_types):
    #import the labels from the csv document

    label_table  = pd.read_csv(label_path, sep='\t',header=0)
    # label_table.head
    
    # convert start and duration to seconds
    f = lambda x: get_sec_long(x["Start"])
    label_table["StartSeconds"] = label_table.apply(f, axis=1)
    
    f = lambda x: get_sec_short(x["Duration"])
    label_table["DurationSeconds"] = label_table.apply(f, axis=1)
    
    # add an end time
    f = lambda x: float(x["StartSeconds"]) + float(x["DurationSeconds"])
    label_table["EndSeconds"] = label_table.apply(f, axis=1)
    
    
    # loop through the labels and turn them into a true/false columns
    for true_label in call_types:
        label_table[true_label] = False
        for old_label in call_types[true_label]:
            label_table.loc[label_table['Name'].str.contains(old_label, regex=False, case = False), true_label] = True
    
    
    # Check which columns are in no category
    df = label_table[list(call_types.keys())]
    unclassed = df.apply(lambda row: True if not any(row) else False if True in list(row) else np.nan, axis=1)
    print("These values will be classed as other : " + str(list(label_table["Name"][list(unclassed)])))
    label_table.loc[list(unclassed), "oth"] = True
    
    return label_table


label_table = create_meerkat_table(label_path, call_types)

#######################################################################################
# format into matrix
import numpy as np
import pandas as pd







def create_label_matrix(label_table, spectro, call_types, start, stop):
    
    '''
    '''
    
    timesteps = spectro.shape[1] # find number of columns for matrix
    colnames = np.arange(start=start, stop=stop, step=(stop-start)/timesteps)
    rownames = call_types.keys() # find number of rows
    timesteps_per_second = timesteps / spec_window_size 
    
    # create an empty matrix where each row represents a call type 
    # and each column represents a timestep which matches the spectrogram timesteps
    label_matrix = pd.DataFrame(np.zeros((len(rownames), timesteps)),
                                index = rownames, columns = colnames)
    # make sure there is a noise row 
    label_matrix.loc['noise'] = 1 
    
    # find the labels for the given spectrogram
    mask = (label_table['StartSeconds']> start) & (label_table['EndSeconds'] < stop)
    mask = mask.index[mask==True]
    # probably not the most elegant, but should allow multiple calls
    for calli in mask: #loop over the calls that occur in that spectrogram and find their type
        call_name = (label_table.loc[calli,:] == True)
        call_name = call_name[call_name==True].index
        for calltypei in call_name: #loop through the different types and mark them as 1 and noise as 0 - this allows hybrids
            label_matrix.loc['noise',
                             ((colnames >= float(label_table['StartSeconds'][calli])) & 
                             (colnames <= float(label_table['EndSeconds'][calli])))] = 0    
            label_matrix.loc[label_matrix.index == calltypei,
                                 ((colnames >= float(label_table['StartSeconds'][calli])) & 
                                 (colnames <= float(label_table['EndSeconds'][calli])))] = 1
    
    return label_matrix
    
label_matrix = create_label_matrix(label_table, spectro, call_types, start, stop)


# import numpy as np
# hyena = np.load("/home/kiran/Documents/animal_data_tmp/hyena/rmishra/combined_spec_label/cc16_352a_18001s_2223sto2229sSPEC_LAB.npy")




###################################################

def create_model(x_train, filters, gru_units, dense_neurons, dropout):
    """
    Outputs a non sequntial keras model
    filters = number of filters in each convolutional layer
    gru_units = number of gru units in each recurrent layer
    dense_neurons = number of neurons in the time distributed dense layers
    dropout = dropout rate used throughout the model
    """
    
    
    #KD Let the data drive the dimensions
    # write function to calculate dimesions
    inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))#KD#(259, 64, 1))
    
    c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)
    mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)
    c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)
    mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)
    c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)
    mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

    reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

    # KD time delay recurrent nn - slower but could be replaced at a later stage
    # GRU gaited recurrent network -feeds into future and past - go through each other
    rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)
    rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                              recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)
    
    # KD goes back from flat to spectro
    dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)
    drop_1 = Dropout(rate=dropout)(dense_1)
    dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)
    drop_2 = Dropout(rate=dropout)(dense_2)
    dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)
    drop_3 = Dropout(rate=dropout)(dense_3)
    
    output = TimeDistributed(Dense(9, activation='softmax'))(drop_3)
    model = Model(inp, output)
    return model
