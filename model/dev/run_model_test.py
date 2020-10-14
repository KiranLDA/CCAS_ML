#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:14:25 2020

@author: kiran
"""
import datetime
import pickle

# from network.network_train import NetworkTrain
import numpy as np
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.layers import Input, Flatten
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Activation, SeparableConv2D, concatenate
from keras.layers import Reshape, Permute
from keras.layers import BatchNormalization, TimeDistributed, Dense, Dropout
from keras.models import load_model
from keras.layers import GRU, Bidirectional, GlobalAveragePooling2D

# x_train_aud = xtrain[0]
# x_train_acc_ch2 = xtrain[1]

#assuming the main loop was not yet called, these need to be imported
import numpy as np
import os
train_test_path = '/media/kiran/D0-P1/animal_data/meerkat/preprocessed/'

x_train = np.load(os.path.join(train_test_path, 'x_train.npy'))
x_train = np.asarray(x_train)
x_train = np.expand_dims(x_train, 4)

# x_test = np.load(os.path.join(train_test_path, 'x_test.npy'))
# x_test = np.asarray(x_test)
# x_test = np.expand_dims(x_test, 4)

y_train = np.load(os.path.join(train_test_path, 'y_train.npy'))
y_train = np.asarray(y_train)

# y_test = np.load(os.path.join(train_test_path, 'y_test.npy'))
# y_test = np.asarray(y_test)


#only use a subset for testing purposes
y_data = y_train
x_data = x_train
x_train = x_data[0:15000,:,:,:]
x_test = x_data[10000:11500,:,:,:]
# x_val = x_data[221:230,:,:,:]
y_train = y_data[0:15000,:,:]
y_test = y_data[10000:11500,:,:]
# y_val = y_data[221:230,:,:]


#function parameters
num_calltypes = y_train.shape[2]
filters = 128 #y_train.shape[1] #
gru_units = y_train.shape[1]#128
dense_neurons = 1024
dropout=0.5

# rashmita's version 
epochs = 1
batch_size = 64
# jack's version 
# epochs = 20
# batch_size = 32



inp = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

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

output = TimeDistributed(Dense(num_calltypes, activation='softmax'))(drop_3)
model = Model(inp, output)

# return model

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['binary_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True, verbose=1)
reduce_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.000001)


spectro_list = os.listdir("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/spectrograms")
spectro_list = spectro_list[0:5]
spectro_list = [os.path.join("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/spectrograms", l) for l in spectro_list]


label_list = os.listdir("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_matrix")
label_list = label_list[0:5]
label_list = [os.path.join("/media/kiran/D0-P1/animal_data/meerkat/preprocessed/label_matrix", l) for l in label_list]

import 

generator = batch_generator(spectro_list, label_list, batch = 2)

for data, labels in generator:
    # data = np.asarray(data)
    print(data.shape)
        



# Generator function to produce standardized length training
# sequences for each batch
# generator = PaddedBatchGenerator(examples[train_idx],
#                                 onehotlabels[train_idx],
# #                                 batch_size=batch_size)
# generator = PaddedBatchGenerator([examples[tt] for tt in train_idx],
#                                  onehotlabels[train_idx,:],
#                                  batch_size=batch_size,
#                                  shuffle=True)
# model.fit_generator(generator, steps_per_epoch=steps,
#                     epochs=epochs,
#                     callbacks=[loss, error, tensorboard, confusion],
#                     class_weight=lossweight,
#                     shuffle=False,
#                     validation_data=(testexamples, testlabels))

# have to calculate steps
steps = 

model.fit_generator(generator, steps_per_epoch=steps,
                    epochs = epochs,
                    callbacks = [early_stopping, reduce_lr_plat],
                    # class_weight = lossweight,
                    shuffle = True,
                    validation_data = (x_test, y_test))



# model_fit = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
#                       validation_data=(x_test, y_test), shuffle=True,
#                       callbacks=[early_stopping, reduce_lr_plat])


###### Still need to be edited to save the model
date_time = datetime.datetime.now()
date_now = str(date_time.date())
time_now = str(date_time.time())
sf = "/media/kiran/D0-P1/animal_data/meerkat/saved_models/model_" + date_now + "_" + time_now
if not os.path.isdir(sf):
        os.makedirs(sf)

model.save(sf + '/savedmodel' + '.h5')
with open(sf + '/history.pickle', 'wb') as f:
    pickle.dump(model_fit.history, f)



# plot_accuracy(model_fit, sf)
# plot_loss(model_fit, sf)
# plot_ROC(model, x_val, y_val, sf)
# plot_class_ROC(model, x_val, y_val, sf)
# save_arch(model, sf)
