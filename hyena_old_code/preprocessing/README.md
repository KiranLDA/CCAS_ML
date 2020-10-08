create_spectrograms.py splits typically 2 hour audio files into mel power spectrograms of 6 seconds long with a 3 second 
sliding window. Using a sliding window increases the dataset size as the same call will potentially be in more than 1 clip
but in a different place in time. The mel power spectrograms are created with 64 mel bands, 2048 Fast Fourier transform
window length and with a 512 samples between successive frames (hop length). The resulting spectrogram is a matrix of
values of shape 259 x 64 where 259 is the number of timesteps and 64 is the number of mel bands. Each second is therefore
represented by 43.1667 timesteps.

create_labels.py creates a matrix of 0s with dimensions 8 x 259, 8 for the number of call types and 259 for the number of
timesteps. The matrix is updated with 1s for each relevant timestep and call type position. Each audio file in the dataset
has a corresponding label file. The labels have begin and end times for each call and this information is used as the
basis for creating the labels.
