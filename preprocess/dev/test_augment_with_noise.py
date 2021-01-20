#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 13:38:54 2021

@author: kiran
"""


def augment_with_noise(spec_filepaths, noise_filepaths, wav_filepaths, calltype, scaling_factor, other_ignored_in_training,
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
        noise_filepaths: 
            list of strings - folder where all the spectrograms are stored (can be the same as spec_filepaths or different if noise spectrograms aare storeed elsewhere)
        wav_filepaths:
            string - folder where all the wav files are stored
        calltype:
            string - call type to be augmented e.g. "cc" or "GRN"
        scaling_factor:
            float - might want to scale noise down if it is being added to a call, 
            e.g. scaling_factor = 1 for no scaling or scaling_factor = 0.3 so that noise is scaled to 30%
        other_ignored_in_training:
            True or False whether or not to include other in the training
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
            string which specifies which label any labels which do not fall into the call_types dectionary will be allocated to. 
            For instance, with meerkats, the 'chew' label might be relabelled as "oth". Normally this should be 
            in the call_types dictionary, but if not, this label will be created.
        label_for_noise:
            string used to label bakcground noise e.g. "noise"
    Output:
        augmented_data:
            an array containing the sum of call and the noise files
        augmented_spectrogram:
            a numpy array of a mel spectrogram of the augmented data
        augmented_label:
            a numpy array containing the labels
        aug_spec_filename: 
            filename for the augmented spectrogram (this way it tracks which file it came from and what the start and stop are)
        aug_mat_filename:
            filename for the augmented spectrogram (this way it tracks which file it came from and what the start and stop are)
        
        
    '''
    ##  NEW INPUTS 
    call_table_dict
    wav_filepaths
    mega_table
    wav_filepaths = audio_filepaths
    random_range # = call_offset
    # # randomly choose a spectrogram 
    # call_spec = random.choice([x for x in spec_filepaths if calltype in x])#glob.glob(folder + "/*" + calltype +".npy")
    # # keep only the file name
    # call_spec = os.path.basename(call_spec) 
    # # keep only the general name of the file so it can be linked to the corresponding .wav
    # call_bits = re.split("_SPEC_", call_spec)
    # file_ID = call_bits[0] 
    calltype = "sn"
    index = 1
    file_ID = call_table_dict[calltype].iloc[index]["File"]
    # find the wav
    call_wav_path = [s for s in wav_filepaths if file_ID in s][0]
    #load the wave
    y, sr = librosa.load(call_wav_path, sr=None, mono=False)
    
    
    # find the corresponding labels
    label_table = mega_table.loc[mega_table["File"]==file_ID]
    #save the label tables with other, but for the purpose of labelling, remove other
    if other_ignored_in_training:
        label_table = label_table[label_table[label_for_other] == False]
        label_table= label_table.reset_index(drop=True)
    
    if sample_size.loc[sample_size["label"] == calltype, "data_augment"] >= 1:
        for i in range(sample_size.loc[sample_size["label"] == calltype, "data_augment"]):
    
            #randomise the start a little so the new spectrogram will be a little different from the old
            call_start = round(float(call_table_dict[calltype].iloc[index]["Start"]+np.random.uniform(-random_range, random_range, 1)), 3)
            call_stop = round(call_start + spec_window_size,3 )
            
            start_lab = int(round(sr * decimal.Decimal(call_start),3))
            stop_lab =  int(round(sr * decimal.Decimal(call_stop),3))
            #suset the wav
            data_subset = np.asfortranarray(y[start_lab:stop_lab])
            
    
            # randomly choose a noise file - same as above, only with noise
            noise_event =  mega_noise_table.sample()#random.choice([x for x in noise_filepaths if label_for_noise in x])#glob.glob(folder + "/*" + calltype +".npy")
            noise_ID = noise_event.iloc[0]["File"]
            noise_wav_path = [s for s in wav_filepaths if noise_ID in s][0]
            noise_start = round(float(np.random.uniform(noise_event.iloc[0]["Start"], noise_event.iloc[0]["End"], 1)), 3)
            noise_stop = round(noise_start + spec_window_size,3 )
            y_noise, sr = librosa.load(noise_wav_path, sr=None, mono=False)
            start = int(round(sr * decimal.Decimal(noise_start),3))
            stop =  int(round(sr * decimal.Decimal(noise_stop),3))
            noise_subset = np.asfortranarray(y_noise[start:stop])
    
    # combine the two
    augmented_data = data_subset + noise_subset * scaling_factor
    # generate spectrogram
    augmented_spectrogram = generate_mel_spectrogram(augmented_data, sr, 0, spec_window_size, 
                                                      n_mels, window, fft_win , fft_hop , normalise)
    # generate label
    augmented_label = create_label_matrix(label_table, augmented_spectrogram,
                                          call_types, call_start, call_stop, 
                                          label_for_noise)
    
    # find out what the label is for this given window so that later we can choose the label/test set in a balanced way
    file_label = list(augmented_label.index.values[augmented_label.where(augmented_label > 0).sum(1) > 1])
    if len(file_label) > 1 and label_for_noise in file_label:
        file_label.remove(label_for_noise)
    category = '_'.join(file_label)
            
    # Save these files
    aug_spec_filename = file_ID + "_SPEC_" + str(call_start) + "s-" + str(call_stop) + "s_NOISE_AUGMENTED_" + category + ".npy"
    aug_mat_filename = file_ID + "_MAT_" + str(call_start) + "s-" + str(call_stop) + "s_NOISE_AUGMENTED_" + category + ".npy"
        
    
    return augmented_data, augmented_spectrogram, augmented_label, aug_spec_filename, aug_mat_filename

