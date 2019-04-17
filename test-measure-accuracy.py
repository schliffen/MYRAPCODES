#
# The goal is to test the accuracy of the trained models
# Ali Nehrani for RAPSODO
#
import os,sys
import numpy as np
from numpy import zeros
from numpy.random  import randint
from numpy.random import random
from matplotlib import pyplot
import _pickle as pickle
import argparse
import keras
import cv2
from keras.models import model_from_json
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import coremltools
# load data generator
# from video_seq_generation import data_generator
#
# testing model on the other data set
# arguments
argp = argparse.ArgumentParser()
argp.add_argument('-d', '--ddir', default='/home/ali/BASKETBALL_DATA/test/',
                  help='direction of testing data')
argp.add_argument('-dn', '--dat', default='video_ba_03_data-01.pickle', help='direction of testing data')
argp.add_argument('-dl', '--lbl', default='video_ba_03_label-01.pickle', help='direction of testing label')
argp.add_argument('-m', '--mdl', default='./models/mobnet_in-out_01', help='direction of testing data')
argp.add_argument('-c', '--cls', default=['ball', 'noball'], help='class names')
args = argp.parse_args()

# this is for pickle data
# with open(args.ddir + args.dat, 'rb') as f:
#     gold_data = pickle.load(f)
# with open(args.ddir + args.lbl, 'rb') as f:
#     gold_label = pickle.load(f)
# trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, random_state=3, shuffle=True, train_size=.8)
# del gold_data, gold_label
#
# loading data from directory
seed = 30

def img_data_gen(args):
    data_list_1 = os.listdir(args.ddir + args.cls[0] + '/')
    data_list_2 = os.listdir(args.ddir + args.cls[1] + '/')
    # random selectio of images
    ind_1 = randint(0, len(data_list_1), 100)
    ind_2 = randint(0, len(data_list_2), 100)

    img_data =[]
    img_label =[]
    for ind in ind_1:
        img = cv2.imread(args.ddir +  args.cls[0] + '/' + data_list_1[ind])
        img_data.append(img)
        img_label.append(1)
    for ind in ind_2:
        img = cv2.imread(args.ddir +  args.cls[1] + '/' + data_list_2[ind])
        img_data.append(img)
        img_label.append(0)
    return img_data, img_label


# collecting test data
# prediction with trained model

# loadin the model
# Serializing model to jason
js_file = open(args.mdl + '.json','r')
loaded_json_model = js_file.read()
js_file.close()
# saving the loaded model as coreml
from keras.applications import mobilenet
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json
# import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# mobilenet.mobilenet.
# ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
with CustomObjectScope({'relu6': mobilenet.mobilenet.relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights(args.mdl + '.h5')
#
# accuracy measure for images list
def il_accuracy(model, data, label):
    tp = 0; tn = 0
    fp = 0; fn = 0
    for j in range(len(data)):
        try:
            img = cv2.resize(data[j], (68, 68))
            pred = model.predict(np.expand_dims(img,0))
            print('prediction:', pred )
            print('gtrue: ', label[j])
            if pred.argmax() == label[j]:
                if label[j] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if label[j] == 1:
                    fp += 1
                else:
                    fn += 1
        except:
            print('blank data')
    usual = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fn)
    recall = tp/(tp + fp)
    print('true positive:', tp)
    print('true negative:', tn)
    print('false positive:', fp)
    print('false negative:', fn)
    return precision, recall, 2* precision/(precision + recall), usual



# computing prediction accuracy
def m_accuracy(model, tX, tY):
    tp = 0; tn = 0
    fp = 0; fn = 0
    for j in range(len(tX)):
        # extracting and reshaping the video
        shot_x = []

        for i in range(len(tX[j])): # trX.shape[1]
            img = tX[j][i].astype(np.float32)
            # img = img.reshape(28,28,3)
            shot_x.append(img.flatten())
        shot_x = np.array(shot_x)
        # creating batch data and label

        if tY[j] ==1:
            shot_y = np.array([0, 1])
        elif tY[j] == 0:
            shot_y = np.array([1, 0])
        try:
            pred = model.predict(np.expand_dims(shot_x,0))
            # print('prediction:', pred )
            # print('gtrue: ', tY[j])
            if pred.argmax() == tY[j]:
                if tY[j] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if tY[j] == 1:
                    fp += 1
                else:
                    fn += 1
        except:
            print('blank data')
    usual = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fn)
    recall = tp/(tp + fp)
    print('true positive:', tp)
    print('true negative:', tn)
    print('false positive:', fp)
    print('false negative:', fn)
    return precision, recall, 2* precision/(precision + recall), usual

# pre, rec, F, total = m_accuracy(loaded_model, trX, trY)

imd_inp, img_lbl = img_data_gen(args)
pre, rec, F, total = il_accuracy(loaded_model, imd_inp, img_lbl)


# print('Number of testing data: ', len(trX))
print('total accuracy: ', total)
print('precision: ', pre)
print('recall: ', rec)
print('F measure: ', F)


# using sklearn accuracy measure
# we can also use
#
# result = model.predict_classes(seq1X, batch_size=1, verbose=0)
