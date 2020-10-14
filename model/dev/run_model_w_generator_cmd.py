import ntpath

import os

from glob import glob

from itertools import compress

label_dirs = ["/home/kiran/Documents/ML/meerkats_2017/labels_CSV", #2017 meerkats
            "/home/kiran/Documents/ML/meerkats_2019/labels_csv"]
audio_dirs= ["/media/kiran/Kiran Meerkat/Kalahari2017",
             "/media/kiran/Kiran Meerkat/Meerkat data 2019"]

save_spec_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/train_data/spectrograms'

save_mat_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/train_data/label_matrix'

save_label_table_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_table'

train_test_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'

skipped_files =[]

for diri in [save_spec_path, save_mat_path , save_label_table_path]:
    if not os.path.exists(diri):
        os.mkdir(diri)

EXT = "*.csv"

label_filepaths = []

for PATH in label_dirs:
    label_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])


audio_filepaths = []

EXT = "*.wav"

for PATH in audio_dirs:
    audio_filepaths.extend( [file for path, subdir, files in os.walk(PATH) for file in glob(os.path.join(path, EXT))])

audio_filepaths = list(compress(audio_filepaths, ["SOUNDFOC" not in filei for filei in audio_filepaths]))

audio_filepaths = list(compress(audio_filepaths, ["PROCESSED" not in filei for filei in audio_filepaths]))

audio_filepaths = list(compress(audio_filepaths, ["LABEL" not in filei for filei in audio_filepaths]))

audio_filepaths = list(compress(audio_filepaths, ["label" not in filei for filei in audio_filepaths]))

audio_filepaths = list(compress(audio_filepaths, ["_SS" not in filei for filei in audio_filepaths]))

audio_filenames = [os.path.splitext(ntpath.basename(wavi))[0] for wavi in audio_filepaths]

label_filenames = []

for filepathi in label_filepaths:
    for audio_nami in audio_filenames:
        if audio_nami in filepathi:
            label_filenames.append(audio_nami)

 

# label_filenames = label_filenames[0:6]

import train_test

split_data = train_test.train_test(label_filenames,  0.75, save_spec_path, save_mat_path) 

x_train_filelist, y_train_filelist, x_val_filelist, y_val_filelist, x_test_filelist, y_test_filelist =  split_data.randomise_train_val_test()  

#x_train_filelist, y_train_filelist, x_test_filelist, y_test_filelist =  split_data.randomise_train_val_test()  
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#       CREATE THE TRAINING AND VALIDATION DATASETS FOR TRAINING RNN
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------


# save_spec_path = os.path.join(train_path + "spectrograms")
# save_mat_path = os.path.join(train_path + "label_matrix")

# file_list = label_filenames
# shuffle(file_list)
        
# # randomly divide the files into those in the training, validation and test datasets
# split_index = floor(len(file_list) * split)
# training = file_list[:split_index]
# validation = file_list[split_index:]

        
# ### TRAINING FILES for data generator
# # Get the spectrogram dataset
# spec_npy_filelist = os.listdir(save_spec_path)
# x_train_filelist = []
# for training_file in training:
#     for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
#         x_train_filelist.append(save_spec_path + '/' + spectro_npy)


# # get the label dataset
# mat_npy_filelist = os.listdir(save_mat_path)
# y_train_filelist = []
# for training_file in training:
#     for mat_npy in [s for s in mat_npy_filelist if training_file  in s]:
#         y_train_filelist.append(save_mat_path + '/' + mat_npy)

        

# ## VALIDATION FILES for data generator
# # Get the spectrogram dataset
# spec_npy_filelist = os.listdir(save_spec_path)
# x_val_filelist = []
# for training_file in validation:
#     for spectro_npy in [s for s in spec_npy_filelist if training_file  in s]:
#         x_val_filelist.append(save_spec_path + '/' + spectro_npy)


# # get the label dataset
# mat_npy_filelist = os.listdir(save_mat_path)
# y_val_filelist = []
# for training_file in validation:
#     for mat_npy in [s for s in mat_npy_filelist if training_file  in s]:
#         y_val_filelist.append(save_mat_path + '/' + mat_npy)
        


# len(x_train_filelist)
# len(x_val_filelist)
# len(x_test_filelist)

# from batch_generator import Batch_Generator
import model.batch_generator as bg

batch = 32

epochs = 16 # 1 works

train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, False)
val_generator = bg.Batch_Generator(x_val_filelist, y_val_filelist, batch, False)
# test_generator = bg.Batch_Generator(x_test_filelist, y_test_filelist, batch, False)

# len(x_train_filelist)/batch
# 
# import importlib
# importlib.reload(Batch_Generator)

# # b = train_generator 
# steps = b.steps_per_epoch() * 3
# print(steps)
# for i in range(steps):
#     print("epoch %d step %d"%(i // b.steps_per_epoch(), i))
#     [dat, lab] = b.__next__()


# for i in range(906, steps):
#     try:
#         [dat, lab] = b.__next__()
#     except StopIteration as err:
#         pass




import datetime

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
'''
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from tensorflow.keras.layers import Reshape, Permute
from tensorflow.keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, Bidirectional, GlobalAveragePooling2D
'''

x_train, y_train = next(train_generator)

# do it again so that the batch is different

train_generator = bg.Batch_Generator(x_train_filelist, y_train_filelist, batch, False)

num_calltypes = y_train.shape[2]

filters = 128 #y_train.shape[1] #

gru_units = y_train.shape[1] #128

dense_neurons = 1024

dropout = 0.5


inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

c_1 = Conv2D(filters, (3,3), padding='same', activation='relu')(inp)

mp_1 = MaxPooling2D(pool_size=(1,5))(c_1)

c_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_1)

mp_2 = MaxPooling2D(pool_size=(1,2))(c_2)

c_3 = Conv2D(filters, (3,3), padding='same', activation='relu')(mp_2)

mp_3 = MaxPooling2D(pool_size=(1,2))(c_3)

reshape_1 = Reshape((x_train.shape[-3], -1))(mp_3)

rnn_1 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(reshape_1)

rnn_2 = Bidirectional(GRU(units=gru_units, activation='tanh', dropout=dropout, 
                          recurrent_dropout=dropout, return_sequences=True), merge_mode='mul')(rnn_1)

dense_1  = TimeDistributed(Dense(dense_neurons, activation='relu'))(rnn_2)

drop_1 = Dropout(rate=dropout)(dense_1)

dense_2 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_1)

drop_2 = Dropout(rate=dropout)(dense_2)

dense_3 = TimeDistributed(Dense(dense_neurons, activation='relu'))(drop_2)

drop_3 = Dropout(rate=dropout)(dense_3)

output = TimeDistributed(Dense(num_calltypes, activation='softmax'))(drop_3)

RNN_model = Model(inp, output)

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

RNN_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)

reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)

'''
RNN_model.fit_generator(train_generator, 
                    steps_per_epoch = train_generator.steps_per_epoch(),
                    epochs = epochs,
                    callbacks = [early_stopping, reduce_lr_plat])

'''
RNN_model.fit_generator(train_generator, 
                        steps_per_epoch = train_generator.steps_per_epoch(),
                        epochs = epochs,
                        callbacks = [early_stopping, reduce_lr_plat],
                        # shuffle = True,
                        validation_data = val_generator,
                        validation_steps = val_generator.steps_per_epoch() )



date_time = datetime.datetime.now()

date_now = str(date_time.date())

time_now = str(date_time.time())

sf = "/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_shuffle_test_" + date_now + "_" + time_now

if not os.path.isdir(sf):
        os.makedirs(sf)

RNN_model.save(sf + '/savedmodel' + '.h5')




