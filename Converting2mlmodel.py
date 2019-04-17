#
# The Goal of this file is to convert the model into mlmodel to be used in ios
## Ali Nehrani for RAPSODO
import os, sys
import tensorflow as tf
import matplotlib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import coremltools
import keras
# fix random seed for reproductability
seed = 3
np.random.seed(seed)
#
# --------------------------------------------------------------------------------------
#
argp = argparse.ArgumentParser()
argp.add_argument('-d', '--ddir', type=str, default='/home/ali/CLionProjects/in_out_detection/models/', help='json model directory')
argp.add_argument('-n', '--mdl', type=str, default='mobnet_in-out_400', help='the name of the saved models',)
argp.add_argument('-s', '--sdir', type=str, default='/home/rapsodo/CLionProjects/in_out_detection/mlmodel/', help='saving data directory')
#
args = argp.parse_args()

#
# digits = load_digits()
#mobnet_in-out_02
# model.save('complete_model.h5')
# # now the turn is loading the model and creating it
# print("[INFO] loading model...")
# model = load_model(args["model"])

# Serializing model to jason
js_file = open(args.ddir + args.mdl +'.json','r')
loaded_json_model = js_file.read()
js_file.close()
# loaded_model = model_from_json(loaded_json_model)
#
# # loading weights to the new model
# # loaded_model.load_weights('model_sl01_w.h5')
# #
#
# from  keras.activations import relu
# def relu6(x):
#     return relu(x,max_value=6)

# keras.activations.relu(x,max_value=6)
from keras.applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json

with CustomObjectScope({'relu6': relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights(args.ddir + args.mdl + '.h5')
#
coreml_model = coremltools.converters.keras.convert(loaded_model,
                                                    input_names="image",
                                                    image_input_names="image"
                                                    )
coreml_model.save('mobnet_ball_400.mlmodel')

print('model is saved successfully!')
# ----------------------------------------------------------------
