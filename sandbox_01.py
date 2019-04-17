#
# Combining  Keras models
#
import os,sys
import numpy as np
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
# from keras_applications.mobilenet import MobileNet
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Input, Reshape
from keras.models import model_from_json
from keras.losses import mean_squared_error
import coremltools
import argparse
argp = argparse.ArgumentParser()
argp.add_argument('-d', '--ddir', type=str, default='./models/', help='json model directory')
argp.add_argument('-n', '--mdl', type=str, default='combined_model', help='the name of the saved models',)
argp.add_argument('-s', '--sdir', type=str, default='./mlmodel/', help='saving data directory')
#
args = argp.parse_args()



# generating sample models
model1 = Sequential()
model1.add(Dense(128, input_shape=(26,26)))
model1.add(Dense(10))
model1.compile(optimizer='adam', loss=mean_squared_error, metrics = ['accuracy'])
model1.summary()

model2 = Sequential()
model2.add(Dense(128, input_shape=(26,26)))
model2.add(Dense(10))
model2.compile(optimizer='adam', loss=mean_squared_error, metrics = ['accuracy'])
model2.summary()

model3 = Model(inputs =[model1.input, model2.input], outputs = [model1.output, model2.output])
model3.compile(optimizer='adam', loss=mean_squared_error, metrics = ['accuracy'])
# combining two models
model3.summary()
# -------------------------------------------------------------
# model input - second and flexible approach
input = Input((26,26,3))
out1 = Dense(128, input_shape=(26,26,3))(input)
out1 = Dense(10)(out1)

out2 = Dense(512, input_shape=(26,26,3))(input)
out2 = Dropout(.5)(out2)
out2 = Dense(2)(out2)


model4 = Model(inputs =input, outputs = [out1, out2])

model4.compile(optimizer='adam', loss=mean_squared_error, metrics = ['accuracy'])
# combining two models
model4.summary()


# saving the model
model_json = model4.to_json()
with open(args.ddir  + args.mdl +  ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model4.save_weights(args.ddir  + args.mdl + "_01.h5")
print("Saved model to disk")


# saving model
js_file = open(args.ddir + args.mdl +'.json','r')
loaded_json_model = js_file.read()
js_file.close()


# loading model
from keras.applications.mobilenet import relu6
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json

with CustomObjectScope({'relu6': relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights(args.ddir + args.mdl + '_01.h5')
#
coreml_model = coremltools.converters.keras.convert(loaded_model,
                                                    input_names="image",
                                                    image_input_names="image"
                                                    )
coreml_model.save(args.sdir + args.mdl +'_coreml.mlmodel')
