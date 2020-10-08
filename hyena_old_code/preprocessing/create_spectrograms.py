import librosa
import librosa.display
import numpy as np
import os


def save_spectrogram(filepath, start, stop, y, sr, path_save):
    """
    Input: .wav filepath, with start and stop time in seconds,
            y audio time series, sr samples rate of audio time
            series
    Output: Save a numpy file of mel spectrogram array of 
            dimension (n_mels, t)
    """
    nfft = int(0.020 * sr)
    hop = int(0.010 * sr)

    s = librosa.feature.melspectrogram(y=y[sr * start:(sr * stop)],
                                       sr=sr, n_mels=64, fmax=sr / 2, n_fft=nfft, hop_length=hop, window='hann', win_length=nfft)
    s_db = librosa.power_to_db(s, ref=np.max)
    if not (os.path.exists(path_save)):
        # create the directory you want to save to
        os.mkdir(path_save)
    file = filepath.split("/")[-1].rsplit("_",maxsplit=1)[0]

    np.save(os.path.join(path_save, file + '_' + str(start) + 'sto' + str(stop) + 's'), s_db)
