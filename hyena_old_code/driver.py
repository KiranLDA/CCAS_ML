import datetime
import os
import numpy as np
import librosa
import shutil
import pickle
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
import tensorflow as tf
from network.network_train import create_model, save_model, plot_accuracy, plot_loss, plot_ROC, plot_class_ROC, \
    save_arch, save_folder, create_save_folder
from preprocessing.create_data_groups import fetch_files_with_numcalls
from preprocessing.create_spectrograms import save_spectrogram
from preprocessing.create_labels import create_label_dataframe, create_label_matrix, find_label
from preprocessing.create_datasets import get_train_val_test, save_data

os.environ["CUDA_VISIBLE_DEVICES"]="2" # select GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

def main():
    base_dir = "/cache/rmishra"
    combined_spec_label = os.path.join(base_dir, "combined_spec_label")
    hyena_data = ['cc16_352a', 'cc16_352b', 'cc16_366a', 'cc16_354a', 'cc16_360a']


    for hyena_recording in hyena_data:

        # Fetch all the audio data with greater than 1 hyena call
        audio_path = os.path.join(base_dir,'cc16_ML',hyena_recording)        # audio_path:'/cache/rmishra/cc16_ML/cc16_352a'
        file_list = fetch_files_with_numcalls(audio_path, 1)

        # Copying audio files for every hyena into <base_dir>/<hyena>_converted/audio/ directory
        save_dir = os.path.abspath(os.path.join(base_dir, hyena_recording + '_converted', 'audio'))     # save_dir:'/cache/rmishra/cc16_352a_converted/audio'
        if not (os.path.exists(save_dir)):
            os.makedirs(save_dir)
        for file in file_list['audio']:
            shutil.copy(os.path.join(audio_path, file), os.path.join(save_dir, file))

        # Processing the audio files with a frame window size of 6 sec, advance of 3 sec, and generating the mel spectrogram
        window_size = 6
        slide = 3
        spec_path = os.path.join(os.path.dirname(save_dir), 'spectro')

        for file in os.listdir(save_dir):
            filepath = os.path.join(save_dir, file)
            y, sr = librosa.load(filepath, sr=None)     # librosa downsamples the audio signal to 22050 if not sr=None
            length = int(len(y) / sr)
            remainder = length % window_size
            for i in range(0, length - remainder - window_size, slide):
                print("Saving spectrogram: "+filepath+" "+str(i)+" to "+str(i+window_size))
                save_spectrogram(filepath, i, i + window_size, y, sr, spec_path)


        # Generate label file corresponding to a spectrogram if there is at least one call present in the window
        for f in os.listdir(spec_path):
            if 'LABEL' not in f:
                label = find_label(f, audio_path)  # We dont have different duration files
                begin_time = int(f.split('_')[-1].split('sto')[0])
                end_time = int(f.split('_')[-1].split('sto')[1].split('s')[0])
                spectro = np.load(os.path.join(spec_path, f))
                timesteps = spectro.shape[1]
                timesteps_per_second = timesteps / window_size
                df = create_label_dataframe(os.path.join(audio_path, label),
                                            begin_time,
                                            end_time,
                                            window_size,
                                            timesteps_per_second)
                label_matrix = create_label_matrix(df, timesteps) # one hot encoding the label information for the audio data in a spec frame

                if 1 in label_matrix[0:8, :]:           # Saving the one hot encoded Label file if there is at least one call present in the 6 sec segment
                    print("Saving file: {}".format(spec_path + f[:-4] + 'LABEL'))
                    np.save(os.path.join(spec_path,f[:-4] + 'LABEL'), label_matrix)
                    np.savetxt(os.path.join(spec_path, f[:-4] + 'LABEL'), label_matrix, delimiter=",")
                    combined = np.array((spectro, label_matrix))
                    if not os.path.exists(combined_spec_label):
                        os.mkdir(combined_spec_label)
                    np.save(os.path.join(combined_spec_label, f[:-4] + 'SPEC_LAB.npy'), combined) # Combine and save the spec and label information


    # Cumulating the combined spectrogram and label information in 1plus_dataset.npy
    SAVE_FILE = '1plus_dataset.npy'
    dataset_path = os.path.join(base_dir, "dataset")
    files = os.listdir(combined_spec_label)

    dataset = []
    for i in range(len(files)):
        current_file = np.load(os.path.join(combined_spec_label, files[i]))
        dataset.append([files[i], current_file])
    dataset = np.array(dataset)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    np.save(os.path.join(dataset_path, SAVE_FILE), dataset)

    # Fetch the train, test and validation data
    train_ratio = 0.75
    val_ratio = 0.15
    train_test_data = get_train_val_test(dataset, train_ratio, val_ratio)
    x_train, y_train, train_files = train_test_data.get("train")
    x_val, y_val, val_files = train_test_data.get("val")
    x_test, y_test, test_files = train_test_data.get("test")

    save_data(x_train, 'x_train.npy', dataset_path)
    save_data(y_train, 'y_train.npy', dataset_path)
    save_data(train_files, 'train_files.npy', dataset_path)
    save_data(x_val, 'x_val.npy', dataset_path)
    save_data(y_val, 'y_val.npy', dataset_path)
    save_data(val_files, 'val_files.npy', dataset_path)
    save_data(x_test, 'x_test.npy', dataset_path)
    save_data(y_test, 'y_test.npy', dataset_path)
    save_data(test_files, 'test_files.npy', dataset_path)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)

    # Train the RCNN model
    model = create_model(x_train, filters=128, gru_units=128, dense_neurons=1024, dropout=0.5)
    print(model.summary())
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])
    epochs = 20
    batch_size = 32
    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
    reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                       mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

    model_fit = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                          validation_data=(x_val, y_val), shuffle=True,
                          callbacks=[early_stopping, reduce_lr_plat])

    # model_fit = load_model('saved_models/model_2019-11-17_03:19:50.898753_network_train/savedmodel.h5')
    #
    # with open('saved_models/model_2019-11-17_03:19:50.898753_network_train/history.pickle', 'rb') as handle:  # loading old history
    #     history = pickle.load(handle)

    date_time = datetime.datetime.now()
    sf = save_folder(date_time)
    create_save_folder(sf)

    model.save(sf + '/savedmodel' + '.h5')
    with open(sf + '/history.pickle', 'wb') as f:
        pickle.dump(model_fit.history, f)

    plot_accuracy(model_fit, sf)
    plot_loss(model_fit, sf)
    plot_ROC(model, x_val, y_val, sf)
    plot_class_ROC(model, x_val, y_val, sf)
    save_arch(model, sf)

if __name__=="__main__":
    main()
