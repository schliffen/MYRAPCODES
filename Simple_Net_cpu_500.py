#
# save and load practice
#
import os, sys
import tensorflow as tf
import matplotlib
# matplotlib.use('Agg')
import argeparse
import matplotlib.pyplot as plt
import numpy as np
import keras
import _pickle as pickle
from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import applications
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
import coremltools

# 
# ----- Saving trained model  as coreml "mlmodel" --------
#

#


# model = creat_model()
#
print('saving training checkpoints as hdf5')
# NOTE: hd just saves the weights!
file_path = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

#
history = model.fit(data_x, trY, validation_split=.1, callbacks=callbacks_list,  epochs=epochs, batch_size=4, verbose=0) #
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

# ----------------------------------------------------------------
