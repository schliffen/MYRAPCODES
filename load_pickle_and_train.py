#
# save and load practice
#
import os, sys
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
#
import matplotlib.pyplot as plt
import numpy as np
import keras
import _pickle as pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
import coremltools
# fix random seed for reproductability
seed = 3
np.random.seed(seed)

# #
# # Reading and splitting the data
# data_dir = '/home/yizhuo/BASKETALLNET/inout_data/aa sorted by shot/'
# data_name = 'video_ba_02_data.pickle'
# label_name = 'video_ba_02_label.pickle'
# with open(data_dir + data_name, 'rb') as f:
#     gold_data = pickle.load(f)
# with open(data_dir + label_name, 'rb') as f:
#     gold_label = pickle.load(f)

# # choosing same number positive and negative
# # 1. numbe rof outs

# trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
#
# total_out=0
# for i in range(gold_label.shape[0]):
#     total_out +=1 if gold_label[i] == 0 else 0
# total_in = gold_label.shape[0] - total_out
# print('total in shots: %2f%% and total out shots: %2f%%' % (total_in, total_out))
# # 2. random selection of data
# inp_data = []
# inp_label = []
# #
# nin = 0
# nout=0
# while (nout < total_out) or (nin < total_out):
#     rind = np.random.randint(0,gold_label.shape[0],1)
#     if gold_label[rind] == 0 and (nout < total_out):
#         nout +=1
#         if len(gold_data[rind][0]) < 40:
#             for j in range(40-len(gold_data[rind][0])):
#                 gold_data[rind][0].insert(0, np.zeros(1024))
#         for j in range(40):
#             inp_data.append(gold_data[rind][0][j])
#         inp_label.append(gold_label[rind][0])
#     if (gold_label[rind] == 1) and (nin < total_out):
#         nin +=1
#         if len(gold_data[rind][0]) < 40:
#             for j in range(40-len(gold_data[rind][0])):
#                 gold_data[rind][0].insert(0, np.zeros(1024))
#         for j in range(40):
#             inp_data.append(gold_data[rind][0][j])
#         inp_label.append(gold_label[rind][0])
#
# del gold_label, gold_data
# # saving this randomly equalized data
# with open('sample_video_data.pickle', 'wb') as f:
#     pickle.dump(np.array(inp_data), f, protocol=4)
# with open('sample_video_label.pickle', 'wb') as f:
#     pickle.dump(np.array(inp_label), f, protocol=4)
#
# print('data %1f label %f' %(len(inp_data), len(inp_label)))

# --------------------------------------------------------------------------------------
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Reading and splitting the data
data_dir = '/home/ali/CLionProjects/p_01/Vid_data/'
data_name = 'video_ba_feat_02_data.pickle'
label_name = 'video_ba_feat_02_label.pickle'
#
# enc = OneHotEncoder()
# arenc = enc.fit(digits.target.reshape(-1,1))
# trY = arenc.transform(trY.reshape(-1,1)).toarray()
# tsY = arenc.transform(tsY.reshape(-1,1)).toarray()
# #
# img_dir
#
# trX = trX.reshape(-1,40)
# tsX = tsX.reshape(-1,40)
#
# creating data for training
#
def data_generator(data_dir, data_name, label_name, batch_size = 8):
    with open(data_dir + data_name, 'rb') as f:
        gold_data = pickle.load(f)
    with open(data_dir + label_name, 'rb') as f:
        gold_label = pickle.load(f)
    # splitting data
    trX, tsX, trY, tsY = train_test_split(gold_data, gold_label)
    del gold_data, gold_label
    # TODO: make labels as vectors

    data_x = []
    data_y = []

    while True:
        for j in range(batch_size):
            shot=[]
            for i in range(len(trX[j])): # trX.shape[1]
                img = trX[j][i].astype(np.float32)
                shot.append(img)
            shot = np.array(shot)
            # creating batch data and label
            data_x.append(shot)
            data_y.append(trY[j])
    yield np.array(data_x), np.array(data_y)



# xx, yy = data_generator(data_dir, data_name, label_name)

# # load mobilenet here and cropping the last layers of it:
# monet = applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')
# # changing the mobilenet
# artif_model = Sequential()
# for num,layer in enumerate(monet.layers[:-4]):
#     # print('layer number: ', num, ' -> ', layer)
#     artif_model.add(layer)
#
# artif_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# artif_model.summary()



input = Input((7,7, 1024))
# model 1 the convolution layer

# for i in range(trX.shape[0]):

cnn_model = Sequential()
cnn_model.add(Conv2D(1024, (3, 3), strides=(2, 2), padding='same',
                 use_bias=False, input_shape=(7, 7, 1024)))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid',
                          depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid',
                          depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(ZeroPadding2D(padding=(1, 1)))
cnn_model.add(DepthwiseConv2D((3, 3), padding='valid',
                          depth_multiplier=1, strides=(1, 1), use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(Conv2D(512, (1, 1), strides=(2, 1), padding='same', use_bias=False))
cnn_model.add(BatchNormalization())
cnn_model.add(Activation('relu'))
cnn_model.add(GlobalAveragePooling2D(data_format='channels_last'))

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn_model.summary()



# # creating data for training
# for j in range(trX.shape[0]):
#     for i in range(trX.shape[1]):
#         img = tf.expand_dims(trX[j][i].reshape(224,224,3).astype(np.float32), 0)
#
#         out1 = artif_model(img)

# ZeroPadding2D(padding=(1, 1))
# DepthwiseConv2D((3, 3), padding='valid',
#                           depth_multiplier=1, strides=(2, 2), use_bias=False)
# BatchNormalization()
# Activation('relu')
# Conv2D(1024, (1, 1), strides=(1, 1), padding='same', use_bias=False)
# BatchNormalization()
# Activation('relu')
#
# # the model

out2 = cnn_model(input)

input2 = RepeatVector(40)(out2)


out = Bidirectional(LSTM(units=512, input_shape=(None, 512), return_sequences=True,
                         kernel_initializer="uniform", activation='relu',
                         dropout=.2, forget_bias_init='one',
                         inner_activation='sigmoid', go_backwards=True))(input2)
out =  Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="uniform",
                          activation='relu', go_backwards=True,
                          forget_bias_init='one', inner_activation='sigmoid', dropout=.2))(out)
out =  Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer="uniform",
                          dropout=.2, forget_bias_init='one', inner_activation='sigmoid',
                          activation='relu', go_backwards=True))(out)
out = LSTM(units=50, input_shape=(None, 1024), return_sequences=False, kernel_initializer="uniform",
           forget_bias_init='one', inner_activation='sigmoid',
           dropout=.2, activation='relu')(out)
out = Dropout(rate=.4)(out)
out = Dense(100, kernel_initializer="uniform", activation='tanh')(out)
out = Dropout(rate=.4)(out)
output = Dense(1, kernel_initializer="uniform", activation='softmax')(out)
#
model = Model(input, output)
# #
# compiling the model/
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#
# model = creat_model()
#
print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!
file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
# hyperparameters
from keras.optimizers import SGD
epochs = 50
learning_rate = 10
decay_rate = learning_rate/epochs
momentum = .99
sgd = SGD(lr = learning_rate, momentum=momentum, decay= decay_rate, nesterov= True)
#
# compiling the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# evaluating with the model
model.summary()


#
with tf.device('/GPU:0'):
    history = model.fit(trX, trY, validation_split=.1, callbacks=callbacks_list,  epochs=epochs, batch_size=4, verbose=0) #
# evaluation
scores = model.evaluate(tsX, tsY, verbose=0) #
# printing
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Serializing model to jason
model_jason = model.to_json()
with open('model_train_01.json', 'w') as j_file:
    j_file.write(model_jason)
# Serialize weights to HDF5
model.save_weights('model_train_01_w.h5')
# #
# model.save('complete_model.h5')
# # now the turn is loading the model and creating it
# print("[INFO] loading model...")
# model = load_model(args["model"])

# Serializing model to jason
js_file = open('model_train_01.json','r')
loaded_json_model = js_file.read()
js_file.close()
# loaded_model = model_from_json(loaded_json_model)
#
# # loading weights to the new model
# # loaded_model.load_weights('model_sl01_w.h5')
# #
#
#
from keras.applications.mobilenet import mobilenet
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json
# import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# mobilenet.mobilenet.

# ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
with CustomObjectScope({'relu6': mobilenet.mobilenet.relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights('model_train_01_w.h5')
#

coreml_model = coremltools.converters.keras.convert(loaded_model)
coreml_model.save('coreml_train_01.mlmodel')
# -----------------------------------------------------------------------------
# # another sample method
# model = Sequential()
# left = Sequential()
# left.add(LSTM(output_dim=256, init='uniform', inner_init='uniform',
#               forget_bias_init='one', return_sequences=True, activation='tanh',
#               inner_activation='sigmoid', input_shape=(32, 32)))
# right = Sequential()
# right.add(LSTM(output_dim=256, init='uniform', inner_init='uniform',
#                forget_bias_init='one', return_sequences=True, activation='tanh',
#                inner_activation='sigmoid', input_shape=(32, 32), go_backwards=True))
#
# model.add(Merge([left, right],'sum'))
# ----------------------------------------------------------------
#
# def creat_model():
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units=1024, input_shape=(40, 1024), return_sequences=True,
#                              kernel_initializer="uniform", activation='relu',
#                              dropout=.2, forget_bias_init='one',
#                              inner_activation='sigmoid', go_backwards=True)))
#     model.add(Bidirectional(LSTM(units=512, return_sequences=True, kernel_initializer="uniform",
#                               activation='relu', go_backwards=True,
#                               forget_bias_init='one', inner_activation='sigmoid', dropout=.2)))
#     model.add(Bidirectional(LSTM(units=256, return_sequences=True, kernel_initializer="uniform",
#                               dropout=.2, forget_bias_init='one', inner_activation='sigmoid',
#                               activation='relu', go_backwards=True)))
#     model.add(LSTM(units=50, input_shape=(128, 1024), return_sequences=False, kernel_initializer="uniform",
#                forget_bias_init='one', inner_activation='sigmoid',
#                dropout=.2, activation='relu'))
#     model.add(Dropout(rate=.4))
#     model.add(Dense(100, kernel_initializer="uniform", activation='tanh'))
#     model.add(Dropout(rate=.4))
#     model.add(Dense(1, kernel_initializer="uniform", activation='softmax'))
#     return model
