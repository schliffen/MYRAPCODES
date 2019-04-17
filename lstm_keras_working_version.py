#
# lstm code 02
#
import os, sys
import matplotlib
matplotlib.use("Agg")
import argparse
import keras
import numpy as np
# from My_Data import sample_data
from sklearn.model_selection import train_test_split
from six.moves import cPickle as pickle
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import model_from_json
from tensorflow.keras.applications import mobilenet
# import keras.utils.multi_gpu_utils

import coremltools

# trying to use gpu





#
ap = argparse.ArgumentParser()
ap.add_argument('-d','--logdir', type=str, default='/home/yizhuo/Documents/deep_learning/inout_classification/myvenv/report/log_dir/', help='direction to tensorboard log')
ap.add_argument('-v','--vidir', type=str, default='/home/yizhuo/BASKETALLNET/Vid_data/', help='direction to video files')
ap.add_argument('-s','--source', default='/home/ali/CLionProjects/p_01/Vid_data/',   help='Path to the source metadata file')
ap.add_argument('--checkpoint', default='/home/yizhuo/Documents/deep_learning/inout_classification/myvenv/report/checkpoint/',  help='Path to the model checkpoint')
# ap.add_argument('--stopwords_file', default='/home/ali/CLionProjects/p_01/report/stopwords.txt', help='Path to stopwords file')
ap.add_argument('--summaries_dir', default='/home/yizhuo/Documents/deep_learning/inout_classification/myvenv/report/summaries_dir/', help='Path to stopwords file')
ap.add_argument('-m','--model', default='MobileNet', nargs="?",
                # type=named_model,
                help='Name of the pre-trained model to use')
ag = ap.parse_args()





# download and save keras mobilenet with weights - this works only in keras 2
from keras.applications.mobilenet import MobileNet

model = MobileNet(weights='imagenet', include_top=False, input_shape=(128,128,3))





















nadam = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# defining the model
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(40,1024))) #)
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(256, return_sequences=False))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

print(model.summary())

# data preparation ----
get_data = sample_data()
#
if not os.path.isfile(ag.vidir + 'video_ba_feat_01_data.pickle') and not os.path.isfile(ag.vidir + 'video_ba_feat_01_label.pickle'):
    gold_data, gold_label = get_data.video_data(ag.vidir)
else:
    with open(ag.vidir + 'video_ba_feat_01_data.pickle', 'rb') as f:
                gold_data = pickle.load(f)
    with open(ag.vidir + 'video_ba_feat_01_label.pickle', 'rb') as f:
                gold_label = pickle.load(f)



# splitting the data for training
#
trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, shuffle=True, test_size=.2, random_state=1)
#
# ----------------------------------------------------
#  making all shots with the same size
# TODO: using dynamic model for length variable videos
#
TrX = []
TsX = []
for i in range(trX.shape[0]):
    if len(trX[i]) < 40:
        for j in range(40-len(trX[i])):
            trX[i].insert(0, np.zeros(1024))
    for item in trX[i]:
        TrX.append(item)

TrX = np.array(TrX).reshape(-1,40,1024)

for i in range(tsX.shape[0]):
    if len(tsX[i]) < 40:
        for j in range(40-len(tsX[i])):
            tsX[i].insert(0, np.zeros(1024))
    for item in tsX[i]:
        TsX.append(item)
#
TsX = np.array(TsX).reshape(-1,40,1024)
#



# Train the model
model.fit(TrX, trY, batch_size=8, epochs=2)
# saving the results of the model

# checking the accuracy
loss, acc = model.evaluate(TsX, tsY)
print('model evaluation; accuracy:', acc , 'loss', loss)
# checking visual examplr
rnd = np.random.randint(0,tsY.shape[0]-20,1)
# print('model prediction for some examples: ')
# model.predict_classes(TsX[rnd:rnd+20])

# print('ground truth for those examples: ')
# print(tsY[rnd:rnd+20])

# serialize model to JSON
model_json = model.to_json()
with open("model_202.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_202.h5")
print("Saved model to disk")

# saving the model as coreml model
coreml_model = coremltools.converters.keras.convert(model)

coreml_model.save('model_202.mlmodel')




# later...

## load json and create model
# json_file = open('model_256.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
## load weights into new model
# loaded_model.load_weights("model_256.h5")
# print("Loaded model from disk")

## evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(TsX, trY, verbose=0)





