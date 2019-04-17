#
#  A CNN model for feature extractuin
#
import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import ImageToArrayPreprocessor
# from pyimagesearch.preprocessing import AspectAwarePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from keras import callbacks
#
# from imutils import paths
#
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import random
import coremltools
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Convolution2D, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
from keras.utils.training_utils import multi_gpu_model
import tensorflow
#
from learning_rate import create_lr_schedule
from loss import dice_coef_loss, dice_coef, recall, precision, softmax_loss, custom_loss

# ---------------
from keras.preprocessing.image import  ImageDataGenerator
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                   rotation_range=30, width_shift_range=.3,
                                   height_shift_range=.3, shear_range=.3, zoom_range=.2, horizontal_flip=True,
                                   zca_whitening=True, zca_epsilon=1e-3, fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(rescale=1./255)

args = argparse.ArgumentParser()
args.add_argument("-dt", "--trn_data", default='/home/ali/CLionProjects/rapsodo_ball_classification/data/in_out/train/',
                help="path to train dataset")
args.add_argument("-dv", "--val_data", default='/home/ali/CLionProjects/rapsodo_ball_classification/data/in_out/test/',
                help="path to val dataset")
args.add_argument('-d2', '--savings', default='/home/ali/CLionProjects/in_out_detection/savings/', help='checkpoint path')
ap = args.parse_args()




# Creating model for feature ext + classification
def createModel(input_shape, nClasses =2):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    return model


# ----------------------------------------------------
#
def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)
#
    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x
#
# coreml converter
#
def saveCoreMLModel(kerasModel):
    coreml_model = coremltools.converters.keras.convert(kerasModel,
                                                        input_names=['input'],
                                                        output_names=['probs'],
                                                        image_input_names='input',
                                                        predicted_feature_name='predictedMoney',
                                                        class_labels = 'drive/Resnet/labels.txt')
    coreml_model.save('resnet50custom.mlmodel')
print('CoreML model saved')

#
# model compiler
#def compile_model(compiledModel):
#    compiledModel.compile(loss=keras.losses.categorical_crossentropy,
#                          optimizer=Adadelta(),
#                          metrics=['accuracy'])
#
# Squeez model
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nClasses, activation='softmax'))

out = squeeze_excite_block(input, ratio=16)


# ---------------------------------------------------------
# data generation
train_generator = train_datagen.flow_from_directory(
    ap.trn_data,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
    ap.val_data,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)
#
model1 = createModel(input_shape=(128,128,3))
batch_size = 4
epochs = 600
# model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# weigth_file=ap.savings + "ckpnt/simpleCNN_in-out-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# tbCallBack = TensorBoard(log_dir=ap.savings + 'Graphs/simpleCNN_in-out_600', histogram_freq=0, write_graph=True, write_images=True)
# callbacks_list = [checkpoint,tbCallBack]
#
# another approach for preparation
model1.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    #optimizer=Adam(lr=0.0001),
    loss=softmax_loss,
    metrics=[
        recall,
        precision,
        'accuracy',
        'kullback_leibler_divergence' # I suggest to use: mean_squared_error
    ],
)

lr_base = 0.01 * (float(batch_size) / 16)
# callbacks
scheduler = callbacks.LearningRateScheduler(
    create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
tensorboard = callbacks.TensorBoard(log_dir=ap.savings + 'logs/')
checkpoint = callbacks.ModelCheckpoint(filepath=ap.savings + 'ckpnt/' +  "/simpleCNN_in-out-{epoch:02d}.hdf5",
                                       save_weights_only=True,
                                       period=20
                                       )
callbacks_list = [scheduler, tensorboard, checkpoint]
history = model1.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size ,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    callbacks=callbacks_list,
    verbose=1)

# model1.evaluate(validation_data=validation_generator)

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
