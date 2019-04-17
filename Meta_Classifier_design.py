#
#
# import the necessary packages
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# knn for predicting eye state
# from pandas import read_csv
# from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
# from numpy import mean
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
import _pickle as pickle
from keras.models import Model
from keras import optimizers
# from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout, TimeDistributed, GlobalAveragePooling2D, Input, BatchNormalization, Bidirectional
# from keras.layers import Concatenate, Reshape, Conv2D, Activation, ZeroPadding2D, DepthwiseConv2D, GlobalMaxPooling1D, RepeatVector, MaxPooling2D
# from keras.utils.training_utils import multi_gpu_model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, accuracy_score

# import coremltools
# ---------------
ap = argparse.ArgumentParser()
ap.add_argument("-dt", "--trn_data", default='/home/ali/BASKETBALL_DATA/DEBUG/train/',
                help="path to train dataset")
ap.add_argument("-dv", "--val_data", default='/home/ali/BASKETBALL_DATA/DEBUG/test/',
                help="path to val dataset")
ap.add_argument("-d1", "--fe_data", default='/home/ali/BASKETBALL_DATA/feature_data/',
                help="path to val dataset")

args = ap.parse_args()
# Prefix name coming from p1
unique_prefix = "20181123_100959"

#
# part one: Data augmentation and preparation for meta classifier train
# Description: it was assumed that this gets to matrix of 40 * (2 + 10)
# 40 frames for each of them we have 2 scores and 10 descriptor keypoints
# this problem is kind of multi dimantional alternating temporal event!
#
# Working example for my blog post at:
# http://danijar.com/variable-sequence-lengths-in-tensorflow/
import functools
import sets
import tensorflow as tf
# from  tensorflow.rnn.rnn_cell import GRUCell
from tensorflow.contrib.rnn import GRUCell

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class VariableSequenceClassification:

    def __init__(self, data, target, num_hidden=512, num_layers=2):
        self.data = data
        self.target = target
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        # Recurrent network.
        output, _ = tf.nn.dynamic_rnn(
            GRUCell(self._num_hidden),
            data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        last = self._last_relevant(output, self.length)
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    @staticmethod
    def _last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

class data_process:

    def merge_df_and_select_features(in_df_name, out_df_name, max_len=90):
        # Selected features
        feature_names = ["scores"] # "feat_num_con_ins"
        # Merging the dataframes
        in_df = pd.read_pickle(in_df_name)
        out_df = pd.read_pickle(out_df_name)
        merged_df = pd.concat([in_df, out_df], ignore_index=True)
        merged_df['bin_class'] = np.where(merged_df['class'] == "in", 1, 0)
        X = np.squeeze(merged_df.loc[:, feature_names].values)
        y = merged_df["bin_class"].values.astype(int)

        Y1 = []
        for i in range(len(y)):
           if y[i] == 1:
              Y1.append([1, 0])
           else:
               Y1.append([0, 1])
        data_x = []
        for l in range(X.shape[0]):
            Xt = X[l]
            for j in range(max_len - len(Xt)):
                Xt.append(0)
            data_x.append(Xt)

        X = np.hstack(data_x)

        return np.array(X).reshape(-1,max_len), np.array(Y1)

    def print_all_metrics(y, y_pred):
        # Constructing metrics
        conf_mat = confusion_matrix(y, y_pred)
        acc = accuracy_score(y, y_pred)
        rec = recall_score(y, y_pred)
        prec = precision_score(y, y_pred)

        print("Confusion matrix: ")
        print(conf_mat)
        print("Accuracy: " + str(acc))
        print("Recall: " + str(rec))
        print("Precision: " + str(prec))

    def print_misclassified_shots(y,y_pred,df):

        df_mc = df.iloc[np.where(y != y_pred)][["scenario_id", "shot_id"]]
        df_all = df[["scenario_id", "shot_id"]]

        df_all_table = df_all.groupby(by="scenario_id")["shot_id"].nunique()
        df_mc_table = df_mc.groupby(by="scenario_id")["shot_id"].nunique()
        df_table = pd.concat([df_all_table, df_mc_table],axis=1)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("----------------")
            print("Misclassified shots")
            print(df_mc.sort_values(by=["scenario_id","shot_id"]))
            print("----------------")
            print("Overview")
            print(df_table)


in_df_test   = args.fe_data   + "fv_files_test_" + unique_prefix + "_in.pkl"
out_df_test  = args.fe_data  + "fv_files_test_" + unique_prefix + "_out.pkl"
in_df_train  = args.fe_data   + "fv_files_train_" + unique_prefix + "_in.pkl"
out_df_train = args.fe_data  + "fv_files_train_" + unique_prefix + "_out.pkl"


#-------------------------------
# Constructing classifier
# classifier = LogisticRegression()
classifier = RandomForestClassifier()
#-------------------------------

# Training the data on meta-train set

print("-- TRAIN ---------------------")
# X_train,y_train, df_train = data_process.merge_df_and_select_features(in_df_name, out_df_name)



if __name__ == '__main__':
    # We treat images as sequences of pixel rows.
    X_train, Y_train  = data_process.merge_df_and_select_features(in_df_train, out_df_train)
    X_test, y_test = data_process.merge_df_and_select_features(in_df_test, out_df_test)
    # rows = X_train
    num_classes = Y_train.shape[1]
    # trying to use KNN
    # evaluate knn using 10-fold cross-validation
    scores = list()
    # kfold = KFold(10, shuffle=True, random_state=1)
    # for train_x, train_y in kfold.split(zip(X_train,Y_train)):
    # batch_size = X_train.shape[0]
    #
    # epochs = 10
    # model = KNeighborsClassifier(n_neighbors=3)
    # for i in range(epochs):
    #     for j in range(X_train.shape[0]//batch_size):
    #         rnd_index = np.random.randint(0, X_train.shape[0], batch_size)
    #         batch_x = X_train[rnd_index]
    #         batch_y = Y_train[rnd_index]
    #         # define train/test X/y
    #         # trainX, trainy =
    #         # testX, testy = values[test_ix, :-1], values[test_ix, -1]
    #         # define model
    #         # fit model on train set
    #         model.fit(batch_x, batch_y)
    #         # forecast test set
    #         yhat = model.predict(X_test)
    #         # evaluate predictions
    #         score = accuracy_score(y_test, yhat)
    #         # store
    #         scores.append(score)
    #         print('>%.3f' % score)
# calculate mean score across each run
# print('Final Score: %.3f' % (mean(scores)))




    # ---------------------------tensorflow LSTM MODEL ------------------- variable length -
    max_len = 90
    data = tf.placeholder(tf.float32, [None, max_len, 1])
    target = tf.placeholder(tf.float32, [None, num_classes])
    model = VariableSequenceClassification(data, target)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    batch_size = 64
    for epoch in range(2000):
        for _ in range(X_train.shape[0]//batch_size):
            batch_index = np.random.randint(0, X_train.shape[0], batch_size)
            X_batch = X_train[batch_index]
            Y_batch = Y_train[batch_index]
            sess.run(model.optimize, {data: X_batch.reshape(-1,max_len,1), target: Y_batch})
        error = sess.run(model.error, {data: X_test.reshape(-1,max_len,1), target: y_test})
        print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
