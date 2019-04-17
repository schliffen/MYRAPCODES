#
#
#
# import the necessary packages
import argparse
import keras
import os
import cv2
import random
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import array_to_img, img_to_array, load_img

#
# from pyimagesearch.preprocessing import ImageToArrayPreprocessor
# from pyimagesearch.preprocessing import AspectAwarePreprocessor
# from pyimagesearch.datasets import SimpleDatasetLoader
# from pyimagesearch.nn.conv import MiniVGGNet
# --------------
from keras.optimizers import SGD
from imutils import paths
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# --------------
from keras import optimizers
from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
# --------------
from keras.preprocessing.image import  ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=.1,
                         height_shift_range=.1, shear_range=.2, zoom_range=.2, horizontal_flip=True,
                         fill_mode='nearest', zca_whitening=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
# -------------
ap = argparse.ArgumentParser()
ap.add_argument("-dt", "--trn_data", default='/home/ali/BASKETBALL_DATA/ball_noball/train/',
                help="path to train dataset")
ap.add_argument("-dv", "--val_data", default='/home/ali/BASKETBALL_DATA/ball_noball/validation/',
                help="path to val dataset")

# -------------
args = ap.parse_args()
#
# imagePaths = list(paths.list_images(args.trn_data))
# classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
# classNames = [str(x) for x in np.unique(classNames)]
#
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
#
# importing mobilenet
# this is for plotting lnstant loss

class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        clear_output(wait=True)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.legend()

        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()

        plt.show();

plot = PlotLearning()


mobnet = MobileNet(weights='imagenet', include_top=False, pooling='avg')
#
input = Input((68,68,3))
# small_mobnet = Sequential()
# out1 = mobnet.layers[0](input)

for layer in mobnet.layers[:-5]:
    layer.trainable = False
    # out1 = layer(out1)

for layer in mobnet.layers[-5:]:
    layer.trainable = True
    # out1 = layer(out1)

out1 = mobnet(input)

small_monet = Model(input, out1)
small_monet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
small_monet.summary()

out2 = Dense(512, input_shape=(1, 1024), activation='relu')(out1)
out2 = Dropout(0.6)(out2)
out2 = Dense(2, activation='softmax')(out2)
# out2 = Dense(1, activation='softmax')(out2)
f_model = Model(input, out2)
f_model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
f_model.summary()

#
# def simple_model(train_data, train_labels, validation_data, validation_labels):
# model_2 = Sequential()
# model_2.add(Dense(512, input_shape=(None, None, 512), activation='relu'))
# model_2.add(Dropout(0.6))
# model_2.add(Dense(512, activation='relu'))
# model_2.add(Dropout(0.5))
# model_2.add(Dense(512, activation='relu'))
# model_2.add(Dropout(0.5))
# model_2.add(Dense(2, activation='softmax'))
# print('w')
#

weigth_file="./weights/PreTrain_monet_w-improvement-cputtt-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tbCallBack = TensorBoard(log_dir='./Graph/Pretrain_683monetttt', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tbCallBack]
#

train_generator = train_datagen.flow_from_directory(
    args.trn_data,
    target_size=(68, 68),
    batch_size=32,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    args.val_data,
    target_size=(68, 68),
    batch_size=16,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True)



# Applying crossvalidation in order to optimize the parameters
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# setting random seed for reproductability
np.random.seed(3)
estimator = []
estimator.append(('standardize', StandardScaler))
# estimator.append(('mlp', KerasRegressor(build_fn=f_model, epochs=100, batch_size=5, verbose = 0)))
# pipeline = Pipeline(estimator)
# kfold = KFold(n_splits=10, random_state=3)

# # preparing data for cross validation
# data = train_datagen.flow_from_directory(args.trn_data,
#                            target_size = (68, 68), color_mode = "rgb", #classes = Null,
#                            class_mode = "categorical", batch_size = 32, shuffle = True,
#                            seed = 3, # save_to_dir = NULL, save_prefix = "",
#                            # save_format = "png", follow_links = False, subset = NULL,
#                            interpolation = "nearest")

# load image to array


# results = cross_val_score(pipeline, x, y, cv=kfold)

# print('model validation results: mean %2f%% and variance %2f%% ' %( results.mean(), results.std() ))
'''
# AUC for prediction on validation sample
X_val_sample, val_labels = next(validation_generator)
val_pred = model.predict_proba(X_val_sample)
val_pred = np.reshape(val_pred, val_labels.shape)
val_score_auc = roc_auc_score(val_labels, val_pred)
print ("AUC validation score")
print (val_score_auc)
print ('\n')
'''

# Train the model
history = f_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size ,
    epochs=10,
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
with open("./models/mobnet_in-out_05.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
f_model.save_weights("./models/mobnet_in-out_05.h5")
print("Saved model to disk")

small_monet_json = small_monet.to_json()
with open("./models/small_monet_01.json", "w") as json_file:
    json_file.write(small_monet_json)
# serialize weights to HDF5
small_monet.save_weights("./models/small_monet_01.h5")
print("Saved model to disk")
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