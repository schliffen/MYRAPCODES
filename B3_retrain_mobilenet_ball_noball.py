#
#
#
# import the necessary packages
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from pyimagesearch.preprocessing import ImageToArrayPreprocessor
# from pyimagesearch.preprocessing import AspectAwarePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from pyimagesearch.nn.conv import MiniVGGNet
from keras.optimizers import SGD
# from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import random
from keras.models import Model
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
from keras.utils.training_utils import multi_gpu_model
# ---------------
from keras.preprocessing.image import  ImageDataGenerator
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                        rotation_range=30, width_shift_range=.3,
                        height_shift_range=.3, shear_range=.3, zoom_range=.2, horizontal_flip=True,
                        zca_whitening=True, zca_epsilon=1e-3, fill_mode='nearest'

                                   )
validation_datagen = ImageDataGenerator(rescale=1./255)

ap = argparse.ArgumentParser()
ap.add_argument("-dt", "--trn_data", default='/home/ali/CLionProjects/rapsodo_ball_classification/data/in_out/train/',
                help="path to train dataset")
ap.add_argument("-dv", "--val_data", default='/home/ali/CLionProjects/rapsodo_ball_classification/data/in_out/validation/',
                help="path to val dataset")


args = ap.parse_args()


# imagePaths = list(paths.list_images(args.trn_data))
# classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# classNames = [str(x) for x in np.unique(classNames)]

# gold_data = []
# gold_label =[]
# for item in os.listdir(args["dataset"] + 'in/'):
#     img = cv2.imread(args["dataset"] + 'in/' + item)
#     img = cv2.resize(img,(28,28))
#     gold_data.append(img)
#     gold_label.append(np.array([0,1]))
#
# for item in os.listdir(args["dataset"] + 'out/'):
#     img = cv2.imread(args["dataset"] + 'out/' + item)
#     img = cv2.resize(img,(28,28))
#     gold_data.append(img)
#     gold_label.append(np.array([1,0]))
# seed = 30
# r = random.random()            # randomly generating a real in [0,1)
# random.shuffle(gold_data, lambda : r)  # lambda : r is an unary function which returns r
# random.shuffle(gold_label, lambda : r)


# (trainX, testX, trainY, testY) = train_test_split(gold_data, gold_label, test_size=0.25, random_state=42)


# importing mobilenet
mobnet = MobileNet(weights='imagenet', include_top=False, pooling='avg')

resnet = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=2)

input = Input((128,128,3))
# small_mobnet = Sequential()
# out1 = mobnet.layers[0](input)
#for layer in mobnet.layers[:-7]:
#    layer.trainable = False

for layer in mobnet.layers[-5:]:
    layer.trainable = True
#     out1 = layer(out1)
# optimizers.RMSprop(lr=1e-4)

out1 = mobnet(input)
small_monet = Model(input, out1)
small_monet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
small_monet.summary()

# out2 = Dense(1024, input_shape=(1, 1024), activation='tanh')(out1)
# out2 = Dropout(0.6)(out2)
out2 = Dense(400, input_shape=(1, 1024), activation='relu')(out1)
#out2 = Dropout(0.6)(out2)
#out2 = Dense(80, activation='relu')(out2)
out2 = Dropout(0.4)(out2)
out2 = Dense(2, activation='softmax')(out2)
# out2 = Dense(1, activation='softmax')(out2)
f_model = Model(input, out2)
f_model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['acc'])
f_model.summary()

# def simple_model(train_data, train_labels, validation_data, validation_labels):
# model_2 = Sequential()
# model_2.add(Dense(512, input_shape=(None, None, 512), activation='relu'))
# model_2.add(Dropout(0.6))
# model_2.add(Dense(512, activation='relu'))
# model_2.add(Dropout(0.5))
# model_2.add(Dense(512, activation='relu'))
# model_2.add(Dropout(0.5))
# model_2.add(Dense(2, activation='softmax'))
#print('w')

# modifying the learning rate
learn_rate = .001
momentum = .2
optimizer = SGD(lr=learn_rate, momentum=momentum)


# model for training with gpu
multi_model = multi_gpu_model(f_model, gpus=2)
multi_model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer=optimizer)

weigth_file="./weights/mobnet_in-out-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/mobnet_in-out_600', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]
#

train_generator = train_datagen.flow_from_directory(
    args.trn_data,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
    args.val_data,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=True)


# Train the model
history = f_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size ,
    epochs=600,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size,
    callbacks=callbacks_list,
    verbose=1)

# f_model.fit_generator(aug,
#                       steps_per_epoch=len(gold_data),
#                       # validation_data=test_gen,
#                       # validation_steps=10,
#                       callbacks=callbacks_list,
#                       epochs=30,
#                       shuffle=True,
#                       verbose=1)
# model.reset_states()
#     print('epoch: ', j)
# score = model.evaluate_generator(test_gen, nb_validation_samples/batch_size, workers=12)
# saving the model
model_json = f_model.to_json()
with open("./models/mobnet_in-out_600.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
f_model.save_weights("./models/mobnet_in-out_600.h5")
print("Saved model to disk")

#
# plotting the results for better underestanding the training process
#
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
