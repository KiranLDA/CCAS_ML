#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:30:50 2020

@author: kiran
"""


import os
import glob
import itertools
import random
# find all audio paths (will be longer than label path as not everything is labelled)


import sys
sys.path.append("/home/kiran/Documents/github/meerkat-calltype-classifyer")
# # import my_module


# from preprocess_functions import *
import preprocess.preprocess_functions as pre

audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]


audio_filepaths = []
EXT = "*.wav"
for PATH in audio_dirs:
     audio_filepaths.extend([file for path, subdir, files in os.walk(PATH) for file in glob.glob(os.path.join(path, EXT))])
# get rid of the focal follows (going around behind the meerkat with a microphone)
audio_filepaths = list(itertools.compress(audio_filepaths, ["SOUNDFOC" not in filei for filei in audio_filepaths]))
audio_filepaths = list(itertools.compress(audio_filepaths, ["PROCESSED" not in filei for filei in audio_filepaths]))
audio_filepaths = list(itertools.compress(audio_filepaths, ["LABEL" not in filei for filei in audio_filepaths]))
audio_filepaths = list(itertools.compress(audio_filepaths, ["label" not in filei for filei in audio_filepaths]))
audio_filepaths = list(itertools.compress(audio_filepaths, ["_SS" not in filei for filei in audio_filepaths]))


spectro_folder = "/media/kiran/D0-P1/animal_data/meerkat/preprocessed/train_data/spectrograms"
spec_filepaths = [os.path.join(spectro_folder, x) for x in os.listdir(spectro_folder)] #[x for x in os.listdir(spectro_folder)]
wav_filepaths = audio_filepaths 
calltype="_mo"
noise = "_noise"
random_range=0.1
spec_window_size = 1.
scaling_factor = 0.1
window="hann"

#------------------
# rolling window parameters
spec_window_size = 1
slide = 0.5

#------------------
# fast fourier parameters for mel spectrogram generation
fft_win = 0.01 #0.03
fft_hop = fft_win/2
n_mels = 30 #128
normalise = True

#------------------
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

label_for_other = "oth"
label_for_noise = "noise"
n_steps = -2
stretch_factor = 0.99 #If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
scaling_factor = 0.1
random_range = 0.1
#------------------------
save_label_table_path = "/media/kiran/D0-P1/animal_data/meerkat/new_run_sep_2020/label_table"
# noise aug
augmented_data, augmented_spectrogram, augmented_label, spec_name, label_name = pre.augment_with_noise(spec_filepaths, wav_filepaths, calltype, noise, scaling_factor, 
                                                                                random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
                                                                                save_label_table_path, call_types, label_for_other, label_for_noise)

# pitch shift
augmented_data, augmented_spectrogram, augmented_label, spec_name, label_name = pre.augment_with_pitch_shift(spec_filepaths, wav_filepaths, calltype, n_steps,
                                                                                random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
                                                                                save_label_table_path, call_types, label_for_other, label_for_noise)


# time shft

augmented_data, augmented_spectrogram, augmented_label, spec_name, label_name = pre.augment_with_time_stretch(spec_filepaths, wav_filepaths, calltype, stretch_factor,
                                                                                random_range, spec_window_size, n_mels, window, fft_win, fft_hop, normalise,
                                                                                save_label_table_path, call_types, label_for_other, label_for_noise)



