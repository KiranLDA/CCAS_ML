import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module


# from preprocess_functions import *
import preprocess.preprocess_functions as pre

import random
# import glob
import os
import re
import numpy as np
import decimal
# /media/kiran/D0-P1/animal_data/meerkat/preprocessed/test_data/label_table/HM_VCVM001_HMB_AUDIO_R08_ file_2_(2017_08_03-06_44_59)_ASWMUX221153_LABEL_TABLE.txt


# find all audio paths (will be longer than label path as not everything is labelled)
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]



save_label_table_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_table'
with open(os.path.join(save_label_table_path, "training_files_used.txt")) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
training_filenames = [x.strip() for x in content] 


def augment_with_noise(spec_filepaths, wav_filepaths, calltype, noise, scaling_factor, 
                       random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
                       save_label_table_path, call_types, label_for_other, label_for_noise):
    '''
    This function looks in the folder where the spectrograms are saved, 
    finds a random spectrogram of a specific call type, finds the matching wavfile with a bit of randomness
    find a random noise spectrogram and finds the matching wav with a bit of randomness
    and adds the noise to the call and saves a new spectrogram
    
    Input parameters:
        spec_filepaths: 
            list of strings - folder where all the spectrograms are stored
        wav_filepaths:
            string - folder where all the wav files are stored
        calltype:
            string - call type to be augmented e.g. "cc" or "GRN"
        noise:
            string - how noise is labelled e.g. "noise"
        scaling_factor:
            float - might want to scale noise down if it is being added to a call, 
            e.g. scaling_factor = 1 for no scaling or scaling_factor = 0.3 so that noise is scaled to 30%
        random_range:
            float - want to randomise the chunk of call being augmented so that it is not exactly the same as the originial spcetrogram
            e.g. the size of half if the spectrogram is recommended, so for meerkats this would be half of a second
        spec_window_size:
            float: spectrogram window size in seconds
        n_mels: 
            number of mel bands - suggested 64 or 128
        window: 
            spectrogram window generation type - suggested "hann"
        fft_win: 
            window length (in seconds)
        fft_hop: 
            hop between window starts (in seconds)
        normalise:
            true or false depending on whether we want to normalise accross the 
            mel bands to remove noise and get stronger signal
        save_label_table_path:
            location where label tables are stored
        call_types:
            a dictionary of call types and how they are labelled in the data - it is not case sensitive
        label_for_other:
            string which specifies which label anything that does not fall into the call_types dectionary will be allocated to. 
            For instance, with meerkats, the 'chew' label might be relabelled as "oth" because it is not in the call_types dictionary. 
            Normally an "other" call label should be in the call_types dictionary, but if not, this label will be created.
        label_for_noise:
            string used to label bakcground noise
    Output:
        augmented_data:
            an array containing the sum of call and the noise files
        augmented_spectrogram:
            a numpy array of a mel spectrogram of the augmented data
        augmented_label:
            a numpy array containing the labels
        
    '''
    # for testing
    # spectro_folder = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/train_data/spectrograms"
    # wav_folder = 
    # calltype="_mo"
    # noise = "_noise"
    # random_range=0.3
    # spec_window_size = 1.
    # scaling_factor = 0.3
    
    # randomly choose a spectrogram 
    call_spec = random.choice([x for x in spec_filepaths if calltype in x])#glob.glob(folder + "/*" + calltype +".npy")
    # keep only the file name
    call_spec = os.path.basename(call_spec) 
    # keep only the general name of the file so it can be linked to the corresponding .wav
    call_bits = re.split("_SPEC_", call_spec)
    call_wav = call_bits[0] #+".wav"
    call_label_table = call_bits[0] + "_LABEL_TABLE.txt"
    label_table = os.path.join(save_label_table_path, call_label_table)
    # find the wav
    call_wav_path = [s for s in wav_filepaths if call_wav in s][0]
    #randomise the start a little so the new spectrogram will be a little different from the old
    call_start = round(float(float(re.split("s-", call_bits[1])[0]) + np.random.uniform(-random_range, random_range, 1)), 3)
    call_stop = round(call_start + spec_window_size,3 )
    #load the wave
    y, sr = librosa.load(call_wav_path, sr=None, mono=False)
    start = int(round(sr * decimal.Decimal(call_start),3))
    stop =  int(round(sr * decimal.Decimal(call_stop),3))
    #suset the wav
    data_subset = np.asfortranarray(y[start:stop])
    
    # randomly choose a noise file - same as above, only with noise
    noise_spec = random.choice([x for x in spec_filepaths if noise in x])#glob.glob(folder + "/*" + calltype +".npy")
    noise_spec = os.path.basename(noise_spec) 
    noise_bits = re.split("_SPEC_", noise_spec)
    noise_wav = noise_bits[0] #+".wav"
    noise_wav_path = [s for s in wav_filepaths if noise_wav in s][0]
    noise_start = round(float(float(re.split("s-", noise_bits[1])[0]) + np.random.uniform(-random_range, random_range, 1)), 3)
    noise_stop = round(noise_start + spec_window_size,3 )
    y, sr = librosa.load(noise_wav_path, sr=None, mono=False)
    start = int(round(sr * decimal.Decimal(noise_start),3))
    stop =  int(round(sr * decimal.Decimal(noise_stop),3))
    noise_subset = np.asfortranarray(y[start:stop])
    
    # combine the two
    augmented_data = data_subset + noise_subset * scaling_factor
    augmented_spectrogram = generate_mel_spectrogram(test, sr, 0, spec_window_size, 
                         n_mels, window, fft_win , fft_hop , normalise)
    augmented_label = create_label_matrix(label_table, augmented_spectrogram, 
                                              call_types, start, stop, 
                                              label_for_other, label_for_noise)
    return(augmented_data, augmented_spectrogram, augmented_label)


    
def augment_with_shift( spec_filepaths, wav_filepaths, calltype,
                       n_mels, window, fft_win, fft_hop, normalise):
    # randomly choose a spectrogram 
    call_spec = random.choice([x for x in spec_filepaths if calltype in x])#glob.glob(folder + "/*" + calltype +".npy")
    # keep only the file name
    call_spec = os.path.basename(call_spec) 
    # keep only the general name of the file so it can be linked to the corresponding .wav
    call_bits = re.split("_SPEC_", call_spec)
    call_wav = call_bits[0] #+".wav"
    # find the wav
    call_wav_path = [s for s in wav_filepaths if call_wav in s][0]
    #randomise the start a little so the new spectrogram will be a little different from the old
    call_start = round(float(float(re.split("s-", call_bits[1])[0]) + np.random.uniform(-random_range, random_range, 1)), 3)
    call_stop = round(call_start + spec_window_size,3 )
    #load the wave
    y, sr = librosa.load(call_wav_path, sr=None, mono=False)
    spectrogram = pre.generate_mel_spectrogram(test, sr, call_start, call_stop, 
                         n_mels, window, fft_win , fft_hop , normalise)


def augment_with_stretch:

'''
#-------------------------------------------------------------------------
# /home/kiran/Documents/ML/convolutional-meerkat/call_detector/dev/preprocess

import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module


# from preprocess_functions import *
import preprocess.preprocess_functions as pre

import numpy as np
import librosa
import warnings
import ntpath
import re
import os
from glob import glob
from itertools import chain, compress
from random import random


#----------------------------------------------------------------------------------
# Hyena parameters - for xml later
#----------------------------------------------------------------------------------

# # paths
# audio_path = "/home/kiran/Documents/animal_data_tmp/hyena/rmishra/cc16_ML/cc16_352a/cc16_352a_14401s_audio.wav"
# label_path = "/home/kiran/Documents/animal_data_tmp/hyena/rmishra/cc16_ML/cc16_352a/cc16_352a_14401s_calls.txt"
# spec_window_size = 6 #mk 1
# slide = 3 #mk0.75


# # for spectrogram generation
# i = 3434#260
# start = i
# stop = i + spec_window_size
# fft_win = 0.047
# fft_hop = fft_win/2
# n_mels = 64 

# # for label munging
# sep='\s*\t\s*'
# engine = 'python'
# start_column = "StartTime"
# duration_column = "Duration"
# label_column = "Label"
# convert_to_seconds = False
# label_for_other = "OTH"
# label_for_noise = "NOISE"
# call_types = {    
#     'GIG':["gig"],
#     'SQL':["sql"],
#     'GRL':["grl"],
#     'GRN':["grn", "moo"],
#     'SQT':["sqt"],
#     # 'MOO':["moo"],
#     'RUM':["rum"],
#     'WHP':["whp"],
#     'OTH':["oth"]
#     }




#----------------------------------------------------------------------------------
# Meerkat parameters for xml - later
#----------------------------------------------------------------------------------

#------------------
# File paths
#------------------
label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

#basically the root directory for saving spectrograms and labels
save_data_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'



# Note that the lines below don't need to be modified 
# unless you have a different file structure
# They will create a specific file sub directory pattern in save_data_path

# where to save the label tables
save_label_table_path = os.path.join(save_data_path, 'label_table')

# where to save the training specs and labels
train_path = os.path.join(save_data_path,'train_data')
save_spec_train_path = os.path.join(train_path + "spectrograms")
save_mat_train_path = os.path.join(train_path + "label_matrix")

# where to save the training labels and specs
test_path = os.path.join(save_data_path, 'test_data')
save_spec_test_path = os.path.join(test_path + "spectrograms")
save_mat_test_path = os.path.join(test_path + "label_matrix")


#------------------
# rolling window parameters
spec_window_size = 1
slide = 0.5

#------------------
# fast fourier parameters for mel spectrogram generation
fft_win = 0.01 #0.03
fft_hop = fft_win/2
n_mels = 30 #128

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


# parameters that might be useful later that currently aren't dealt with
# we might want to make a separate category e.g. for focal and non-focal
# 'hyb':["hyb","HYB","hybrid","HYBRID","fu","sq","+"],
# 'ukn':["ukn","unknown","UKN","UNKNOWN"]
# 'nf' :["nf","nonfoc","NONFOC"],
# 'noise':["x","X"]
# 'overlap':"%"
# 'nf':["nf","nonfoc"],'[]
#-----------------------------------------
# Data augmentation parameters
noise_scaling_factor = 0.3


#----------------------------------------------------------------------------------
# Actual bit running the code
#----------------------------------------------------------------------------------

# if any files are skipped because they are problematic, they are put here 
skipped_files =[]


# create the directories for saving the files 
#-----------------------------------------------------------------
for diri in [train_path, test_path , save_label_table_path]:
    if not os.path.exists(diri):
        os.mkdir(diri)


# Find the input data
#-----------------------------------------------------------------

# find all label paths
EXT = "*.csv"
label_filepaths = []
for PATH in label_dirs:
     label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])


# find all audio paths (will be longer than label path as not everything is labelled)
audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
     audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])
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

# Must delete later
# label_filenames = [label_filenames[i] for i in [5,10,15,55,60]]

#----------------------------------------------------------------------------------------------------
# Split the label filenames into training and test files
#----------------------------------------------------------------------------------------------------


from random import shuffle
from math import floor
# split = 0.75
# file_list = label_filenames
# shuffle(file_list)

# # randomly divide the files into those in the training, validation and test datasets
# split_index = floor(len(file_list) * split)
# training_filenames = file_list[:split_index]
# testing_filenames = file_list[split_index:]

# subset for testing purposes
# training_filenames = [training_filenames[i] for i in [1,5,10,15,55]]
# testing_filenames = [testing_filenames[i] for i in [5,2]]


#------------------------------------------------------------------
# because the training and testing filenames are random for each run, 
# lets keep a track record of the ones that were used
#------------------------------------------------------------------
# with open(os.path.join(save_label_table_path, "training_files_used.txt"), "w") as f:
#     for s in training_filenames:
#         f.write(str(s) +"\n")


# with open(os.path.join(save_label_table_path, "testing_files_used.txt"), "w") as f:
#     for s in testing_filenames:
#         f.write(str(s) +"\n")


# load the saved file
with open(os.path.join(save_label_table_path, "training_files_used.txt")) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
training_filenames = [x.strip() for x in content] 

with open(os.path.join(save_label_table_path, "testing_files_used.txt")) as f:
    content = f.readlines()
# remove whitespace characters like `\n` at the end of each line
testing_filenames = [x.strip() for x in content] 


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       TRAINING
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(save_spec_train_path):
    os.mkdir(save_spec_train_path)
if not os.path.exists(save_mat_train_path):
    os.mkdir(save_mat_train_path)


# Start the loop by going over every single labelled file id
for file_ID in training_filenames:
    # file_ID = label_filenames[2]
        
    # find the matching audio for the label data
    audio_path = [s for s in audio_filepaths if file_ID in s][0]
    
    # if there are 2 label files, use the longest one (assuming that the longer one might have been reviewed by 2 people and therefore have 2 set of initials and be longer)
    label_path = max([s for s in label_filepaths if file_ID in s], key=len) #[s for s in label_filepaths if file_ID in s][0]
    
    print ("File being processed : " + label_path)
    
    
    # create a standardised table which contains all the labels of that file - also can be used for validation
    label_table = pre.create_table(label_path, call_types, sep, start_column, duration_column, label_column, convert_to_seconds, 
                                   label_for_other, label_for_noise, engine, True)
    # replace duration of beeps with 0.04 seconds - meerkat particularity
    label_table.loc[label_table["beep"] == True, "Duration"] = 0.04
    
    #save the label_table
    save_label_table_filename = file_ID + "_LABEL_TABLE.txt"

    # Don't run the code if that file has already been processed
    # if os.path.isfile(os.path.join(save_label_table_path, save_label_table_filename)):
    #     continue
    # np.save(os.path.join(save_label_table_path, save_label_table_filename), label_table) 
    label_table.to_csv(os.path.join(save_label_table_path, save_label_table_filename), header=True, index=None, sep=' ', mode='a')
    
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
    
    for rowi in range(len(label_table["Start"])):
        # rowi= 3
        # loop through every labelled call event in the dataset and create a random start time to generate sprectrogram window
        random_start = label_table["Start"][rowi] - (random.random() * spec_window_size)
        
        #don't generate a label for the start stop skipon skipoff
        if label_table["Label"].str.contains('|'.join(label_for_startstop), regex=True, case = False)[rowi]:
            continue
        
        # generate 3 spectrograms for every labelled call 
        #(idea is that it should include either 2 call + 1 noise or 1 call and 2 noise - balances the dataset a little)
        for start in np.arange(random_start, (random_start + (3*slide) ), slide):
            # start = random_start + slide
            stop = start + spec_window_size
            
            
            if stop > label_table["End"][len(label_table["End"])-1]:
                continue
            if start < label_table["Start"][0]:
                continue
            
            
            
            ##########################################
            start = random_start + slide*1
            stop = start + spec_window_size
            
            #import decimal
            start = int(round(sr * decimal.Decimal(start),3))
            stop =  int(round(sr * decimal.Decimal(stop),3))
            data_subset = np.asfortranarray(y[start:stop])
            # data augmentation
            # generate a random number between the end of the last call and the start of this call
            
            noise_range_start = label_table["Start"][rowi-1]
            noise_range_stop = label_table["End"][rowi]

            random_noise_start = round(random.uniform(noise_range_start,noise_range_stop),3)
            random_noise_stop = random_noise_start + spec_window_size
            
            noise_start = int(round(sr * decimal.Decimal(random_noise_start),3))
            noise_stop = int(round(sr * decimal.Decimal(random_noise_stop),3))

            noise_subset = np.asfortranarray(y[noise_start:noise_stop])
            
            augmented_data = data_subset + noise_subset * noise_scaling_factor
            
            
            
            win_length  = int(fft_win * sr) 
            hop_length = int(fft_hop * sr) 
            window='hann'
            
            s = librosa.feature.melspectrogram(y = augmented_data ,
                                       sr = sr, 
                                       n_mels = n_mels , 
                                       fmax = sr/2, 
                                       n_fft = win_length,
                                       hop_length = hop_length, 
                                       window = window, 
                                       win_length = win_length )
            
            spectro = librosa.power_to_db(s, ref=np.max)
            normalise = True
            if normalise:
                spectro = spectro - spectro.mean(axis=0, keepdims=True)
                
            spec_aug = spectro    
                
            start = random_start + slide*1
            stop = start + spec_window_size
                  
            spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                           n_mels = n_mels, window='hann', 
                                           fft_win= fft_win, fft_hop = fft_hop, normalise=True)
            label = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                       stop, label_for_other, label_for_noise) 
            
                        #plot spec
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 12))
            plt.subplot(311)
            librosa.display.specshow(spec_aug, x_axis='time' , y_axis='mel')
            plt.colorbar(format='%+2.0f dB')    
            

            plt.subplot(312)
            librosa.display.specshow(spectro, x_axis='time' , y_axis='mel')
            plt.colorbar(format='%+2.0f dB')  
            
            label_list = list(call_types.keys())
            plt.subplot(313)
            xaxis = range(0, np.flipud(label).shape[1]+1)
            yaxis = range(0, np.flipud(label).shape[0]+1)
            plt.yticks(np.arange(0.5, len(label_list)+0.5 ,1 ),label_list[::-1])
            plt.pcolormesh(xaxis, yaxis, np.flipud(label))
            plt.xlabel('time (s)')
            plt.ylabel('Calltype')
            plt.colorbar(label="Label")
            # plot_matrix(np.flipud(label), zunits='Lab
            #######################################################
      
            spectro = pre.generate_mel_spectrogram(y=y_sub, sr=sr, start=start, stop=stop, 
                                           n_mels = n_mels, window='hann', 
                                           fft_win= fft_win, fft_hop = fft_hop, normalise=True)
             
            
            #generate the label matrix
            label_matrix = pre.create_label_matrix(label_table, spectro, call_types, start, 
                                       stop, label_for_other, label_for_noise)
            
            
            # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
            file_label = list(label_matrix.index.values[label_matrix.where(label_matrix > 0).sum(1) > 1])
            if len(file_label) > 1 and 'noise' in file_label:
                file_label.remove('noise')
            category = '_'.join(file_label)
            
            # Save these files
            save_spec_filename = file_ID + "_SPEC_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            save_mat_filename = file_ID + "_MAT_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            # save_both_filename = file_ID + "_BOTH_" + str(start) + "s-" + str(stop) + "s_" + category + ".npy"
            
            np.save(os.path.join(save_spec_train_path, save_spec_filename), spectro)     
            np.save(os.path.join(save_mat_train_path, save_mat_filename), label_matrix) 
            # np.save(os.path.join(save_spec_path, save_both_filename), np.array((spectro, label_matrix))) 


print(skipped_files)


'''