#
# Training LSTM with multi model gpus
#
import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot
import _pickle as pickle
import matplotlib.pylab as plt
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
import gc
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
# load data generator
from video_seq_generation import data_generator
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional, Convolution2D
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
# ---------------
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
from keras import backend as K

import tensorflow as tf
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type    == 'GPU']
num_gpu = len(get_available_gpus())
print('number of gpus', num_gpu)

# -------------------------------
# mxnet gpu options
# -----


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# config.gpu_options.allow_growth = True
# config.gpu_options.allocator_type = 'BFC'
# sess = tf.Session(config = config)


# arguments
data_dir = '/home/rapsodo/BASKETBALL/V_Data/'
data_name = 'video_ba_02_data.pickle'
label_name = 'video_ba_02_label.pickle'
with open(data_dir + data_name, 'rb') as f:
    gold_data = pickle.load(f)
with open(data_dir + label_name, 'rb') as f:
    gold_label = pickle.load(f)
    # splitting data
trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
del gold_data, gold_label
# TODO: 1 make labels as vectors
#
# TODO: 2 I will apply ball confirmation in order to recognize frames including ball
# generator for LSTM
def data_generator(tX, tY, batch_size = 1):
    data_x = []
    data_y = []
    while True:
        for j in range(batch_size):
            shot=[]
            for i in range(len(tX[j])): # trX.shape[1]
                img = tX[j][i].astype(np.float32)
                shot.append(np.resize(img, (64,64,3)))
            shot = np.array(shot)
            shot = shot.reshape(40,64*64*3)
            # creating batch data and label
            data_x.append(shot)
            if tY[j] ==1:
                data_y.append(np.array([0, 1]))
            else:
                data_y.append(np.array([1, 0]))
        yield np.array(data_x), np.array(data_y)

# ----------------------

# a,b = data_generator(tsX, tsY, batch_size = 8)
#
# A simple network to test the yield system with our data
#
model = Sequential()
model.add(LSTM(50, input_shape=(40, 64*64*3), stateful=False))
# just for cnn
#model.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same',
#                   use_bias=False, input_shape=(40,64*64,3)))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
# just for cnn
#model.add(GlobalAveragePooling2D(data_format='channels_last'))


# hyperparameters
from keras.optimizers import SGD
epochs = 50
learning_rate = 2
decay_rate = learning_rate/epochs
momentum = .8
sgd = SGD(lr = learning_rate, momentum=momentum, decay= decay_rate, nesterov= False)


model.compile(optimizer=sgd, loss='binary_crossentropy',  metrics=['accuracy'])
model.summary()



multi_model = multi_gpu_model(model, gpus=2)

multi_model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='adam')


# fit generator

train_gen = data_generator(trX, trY, batch_size = 1)
test_gen  = data_generator(tsX, tsY, batch_size = 1)

del trX, tsX, trY, tsY
gc.collect()
# model.fit(data, data_y, epochs=2)


n_epochs =50
batch_size = 2

print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!
#file_path = "weights-improvement-{epoch:02d}.hdf5"
#checkpoint = ModelCheckpoint(file_path, verbose=1, save_best_only=True, mode='max')
#callbacks_list = [checkpoint]


weigth_file="./weights/PreTrain_weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Pretrain_128Px', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]






for i in range(n_epochs):
#    with tf.device('/cpu:0'):
     model.fit_generator(generator=train_gen,
                        steps_per_epoch=50,
                        epochs=1,
			callbacks=callbacks_list,
                        # validation_data=test_gen,
                        # validation_steps=10
                        verbose=1)
     # model.reset_states()
     print('epoch number:', i)		


#with tf.device('/cpu:0'):
model.fit_generator(generator=train_gen,
                          steps_per_epoch=50,
                          epochs=1,
                          callbacks=callbacks_list
                          # validation_data=test_gen,
                          # validation_steps=10
                          )

# Serializing model to jason
model_jason = model.to_json()
with open('model_train-50.json', 'w') as j_file:
    j_file.write(model_jason)
# Serialize weights to HDF5
model.save_weights('model_train-50.h5')

# plt.style.use('ggplot')
# ax = plt.figure(figsize=(10, 6)).add_subplot(111)
# ax.plot(x_train[:, 0], label='x_train', color='#111111', alpha=0.8, lw=3)
# ax.plot(y_train[:, 0], label='y_train', color='#E69F00', alpha=1, lw=3)
# ax.plot(model.predict(x_train, batch_size=batch_size)[:, 0],
#         label='Predictions for x_train after %i epochs' % n_epochs,
#         color='#56B4E9', alpha=0.8, lw=3)
# plt.legend(loc='lower right')
