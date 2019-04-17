#
# The goal is to test the accuracy of the trained models
# this testing tool is designed for taking prepared data 
# Ali Nehrani for RAPSODO
#
import numpy as np
from numpy import zeros
from random import randint
from random import random
from matplotlib import pyplot as plt
import _pickle as pickle
import argparse
import keras
import cv2
from keras.models import model_from_json
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# load data generator
# from video_seq_generation import data_generator
#
from keras.preprocessing.image import  ImageDataGenerator
# testing model on the other data set
# arguments
argp = argparse.ArgumentParser()
#<<<<<<< HEAD
argp.add_argument('-d', '--ddir', default='/home/ali/BASKETBALL_DATA/testing_classification', help='direction of testing data')
argp.add_argument('-md', '--mdir', default='/home/ali/CLionProjects/rapsodo_ball_classification/', help='direction of main directory')
argp.add_argument('-m', '--mdl', default='models/mobnet_in-out_002', help='direction of testing data')
argp.add_argument('-l', '--lbl', default= ['in', 'out'])

args = argp.parse_args()

test_datagen = ImageDataGenerator(rescale=1./255)
#
batch_size = 125
#
test_generator = test_datagen.flow_from_directory(
    directory=args.ddir,
    target_size=(128, 128),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)
# prediction with trained model

# loadin the model
# Serializing model to jason
from keras.applications import mobilenet
from keras.utils.generic_utils import CustomObjectScope
from keras.models import model_from_json
# import keras.applications.mobilenet.mobilenet._depthwise_conv_block as DepthwiseConv2D
# mobilenet.mobilenet.
js_file = open(args.mdir + args.mdl + '.json','r')
loaded_json_model = js_file.read()
js_file.close()
# ,'DepthwiseConv2D': mobilenet.DepthwiseConv2D
with CustomObjectScope({'relu6': mobilenet.relu6}):
    loaded_model = model_from_json(loaded_json_model)
    # allmodel = load_model('complete_model.h5')
    # loading weights to the new model
    loaded_model.load_weights(args.mdir + args.mdl + '.h5')

#
# computing prediction accuracy
def m_accuracy(model, tX, tY):
    tp = 0; tn = 0
    fp = 0; fn = 0

    for i in range(len(tX)): # trX.shape[1]//batch_size
        img = tX[i].astype(np.float32)
        img = cv2.resize(img,(128,128))
        try:
            pred = model.predict(np.expand_dims(img,0))

            #if pred.argmax() == tY[i].argmax():
            if pred[0][0] >= .85:
                if tY[i][0] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if tY[i][1] == 1:
                    fp += 1
                else:
                    fn += 1
        except:
            print('blank data')
    usual = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fn + .00001)
    recall = tp/(tp + fp + .00001)
    print('true positive:', tp)
    print('false positive:', fp)
    print('true negative:', tn)
    print('false negative:', fn)

    return precision, recall, (2* precision*recall)/(precision + recall), usual

# loading data for test


for xbatch, ybatch in  test_generator:
    # for i in range(0,9):
    #     # plt.subplot(330 + 1 + i)
    #     # plt.imshow(xbatch[i])
    #     prediction = loaded_model.predict(np.expand_dims(xbatch[i],0))
    #     print('algorithm prediction: ', prediction)
    #     print('ground truth: ', ybatch[i])
    #     if prediction[0][0] > .85:
    #         print('result: in' )
    #     else:
    #         print('result: out')
    #     print('----------------------')
        # plt.title('prediction = ' + str(prediction))
    # plt.show()
    # Confudion matrix

    batch_prediction = loaded_model.predict(xbatch)
    y_true = np.zeros(len(ybatch))
    y_pred = np.zeros(len(ybatch))
    for i in range(len(ybatch)):
        y_true[i] = 1 if ybatch[i][0] else 0
        y_pred[i] = 1 if batch_prediction[i][0] > .85 else 0

    confusion_matrix(y_true= y_true, y_pred=y_pred, labels=[1, 0])
    tp, fp, tn, fn = confusion_matrix(y_true= y_true, y_pred=y_pred, labels=[1, 0]).ravel()
# accuracy measures
#     pre, rec, F, total = m_accuracy(loaded_model, xbatch, ybatch)
    usual = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fn + .00001)
    recall = tp/(tp + fp + .00001)
    F = (2* precision*recall)/(precision + recall + .00001)
    print('---------------------- Report ---------------------------')
    print('Number of testing data: ', len(xbatch))
    print('true positive:', tp)
    print('false positive:', fp)
    print('true negative:', tn)
    print('false negative:', fn)
    print('total accuracy: ', usual)
    print('precision: ', precision)
    print('recall: ', recall)
    print('F measure: ', F)


# using sklearn accuracy measure
# we can also use
#
# result = model.predict_classes(seq1X, batch_size=1, verbose=0)

