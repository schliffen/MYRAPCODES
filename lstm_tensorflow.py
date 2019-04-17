#
# my previous version of LSTM
#
# imports
from subprocess import call
import os, sys, glob
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from six.moves import cPickle as pickle
import keras.utils.multi_gpu_utils
# import _pickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from My_Data import sample_data
# Setting argumant parser
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

# this is developing LSTM without feature extraction in tf
# each input size = input_vec_size=lstm_size=28
# loading the videos and transforming them into the alsoes standard format

# configuration variables
#
input_feature_vec_size = lstm_size = 1024
# time step size is the number of frames taken from a single video
num_classes = 2
time_step_size = 40
batch_size = 10
#
# putting the video together here
#
# cp = cv2.VideoCapture(0)
# print(cp.isOpened())
# 375



#
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# simple many to many lstm design
def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)
    # Make lstm with lstm_size (each input vector size)
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)
    # Linear activation
    # Get the last output
    result = tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat
    return result

# mnist data just for testing the implemenation ----------------------------
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
# trX = trX.reshape(-1, 28, 28)
# teX = teX.reshape(-1, 28, 28)
# ------------------------------------------------------------
# very light dataset for testing
# this part of code will remove to another file as a basic dataset at hand for other processes
# TODO: more efficient data management are required!
#
get_data = sample_data()
#
if not os.path.isfile(ag.vidir + 'video_ba_feat_01_data.pickle') and not os.path.isfile(ag.vidir + 'video_ba_feat_01_label.pickle'):
    gold_data, gold_label = get_data.video_data(ag.vidir)
else:
    with open(ag.vidir + 'video_ba_feat_01_data.pickle', 'rb') as f:
                gold_data = pickle.load(f)
    with open(ag.vidir + 'video_ba_feat_01_label.pickle', 'rb') as f:
                gold_label = pickle.load(f)
#
#
# img_list = os.listdir(ag.source)
# simg = cv2.imread(ag.source + img_list[0])
#
#
# making data prepared for train test split
trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, shuffle=True, test_size=.2, random_state=1)
# setting the test size
# trx = trX[:-7]; tryy = trY[:-7];
# trTg = np.array([item for num,item in enumerate(tryy) if num % time_step_size == 0])
# tsTg = np.array([item for num,item in enumerate(tsY) if num % time_step_size == 0])
# trx = trx.reshape(-1,time_step_size,64)
# tsx = tsX.reshape(-1,time_step_size, 64)
# test_size = tsTg.shape[0]
# one_hot encoding of the target

# shot size unification - adding zero or mean to the beginning of shorter ones
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

TsX = np.array(TsX).reshape(-1,40,1024)


#
# if not os.path.isfile(ag.vidir + 'pos_train_data_01.pickle'):
#     with open(ag.vidir + 'pos_train_data_01.pickle', 'wb') as f:
#                     pickle.dump(TrX,f)
# else:
#     with open(ag.vidir + 'pos_train_data_01.pickle', 'rb') as f:
#                     TrX = pickle.load(f)
#
#
# if not os.path.isfile(ag.vidir + 'pos_test_data_01.pickle'):
#     with open(ag.vidir + 'pos_test_data_01.pickle', 'wb') as f:
#                     pickle.dump(TsX,f)
# else:
#     with open(ag.vidir + 'pos_test_data_01.pickle', 'rb') as f:
#                     TsX = pickle.load(f)
#
# if not os.path.isfile(ag.vidir + 'pos_train_label_01.pickle'):
#     with open(ag.vidir + 'pos_train_label_01.pickle', 'wb') as f:
#                     pickle.dump(trY,f)
# else:
#     with open(ag.vidir + 'pos_train_label_01.pickle', 'rb') as f:
#                     trY = pickle.load(f)
# if not os.path.isfile(ag.vidir + 'pos_test_label_01.pickle'):
#     with open(ag.vidir + 'pos_test_label_01.pickle', 'wb') as f:
#                     pickle.dump(tsY,f)
# else:
#     with open(ag.vidir + 'pos_test_label_01.pickle', 'rb') as f:
#                     tsY = pickle.load(f)
del trX, tsX

# Beginning of the main code ()
X = tf.placeholder("float", [None, time_step_size, input_feature_vec_size])
Y = tf.placeholder("float", [None, num_classes])
#
# get lstm_size and output 10 labels
W = init_weights([lstm_size, num_classes])
B = init_weights([num_classes])
# load lstm model
py_x, state_size = model(X, W, B, lstm_size)
# tf optimization options


py_x, state_size = model(TsX, W, B, lstm_size)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)
#
session_conf = tf.ConfigProto()
session_conf.gpu_options.allow_growth = True
# -----------------------------------------------
# feature extraction part goes here
#
# simg = trX[0].reshape(8,8)
# TODO: I should developed codes for saving weights in order to use thm later!

# cv2.imshow('sample image',simg);
# if cv2.waitKey(0) == ord('q'):
#     cv2.destroyAllWindows()
#
# fext = fextn(ag)
# features = fext.get_feature(simg)
#TODO: loading mobilenet files from the saved file: ./mobilenet_1_0_224_tf_no_top.h
# use tSNE to visualize the features!
# this is important in order to discover correlated features to speed the method up
# fvis = vis('TSNE')
# X_embedded = fvis.named_model().fit_transform(features.reshape(-1,1))
# plt.plot(X_embedded[0],X_embedded[1]); plt.show()

# TODO:makeing the improved
def __loss(losses):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(losses, name='loss')
        tf.summary.scalar('loss', loss)
        return loss
#
# def __train_step(learning_rate, loss):
#     return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#
def __accuracy(predict=py_x, target =Y):
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)
        return accuracy
#
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
# Launch the graph in a session
# introducing checkpoint file
checkpoint_file = '{}/model.ckpt'.format(ag.checkpoint)

#
with tf.Session(config=session_conf) as sess:
    # you need to initialize all variables
    writer = tf.summary.FileWriter(ag.logdir, sess.graph)
    # saver
    saver = tf.train.Saver()
    #
    tf.global_variables_initializer().run()
    tf.summary.merge_all()
    for i in range(100):
        for start, end in zip(range(0, len(trY), batch_size), range(batch_size, len(trY)+1, batch_size)):

            lstm_pred, cost_t = sess.run(predict_op, cost, train_op,  feed_dict={X: TrX[start:end], Y: trY[start:end]})

            # saving checkpoint after certain iterations
            if i % 5 == 0:
                save_path = saver.save(sess, checkpoint_file)
                print('Model saved in: {0}'.format(ag.checkpoint))
                __loss(cost_t)
                print('LSTM prediction: ', lstm_pred)
                print('cost funcstion: ', tf.reduce_mean(cost))


        # TODO: develpoing the argmax results here as well!!



    # for data in test_data:
    #     test_indices = np.arange(len(tsx))  # Get A Test Batch
    #     np.random.shuffle(test_indices)
    #     test_indices = test_indices[0:test_size]
    # preparing the test data
        # computing the accuracy
        test_loss = sess.run(py_x, cost, feed_dict={})
        # measuring the accuracy
        __accuracy(py_x, Y)


        # print(i, np.mean(np.argmax(tsY[test_indices], axis=1) ==
        #                  sess.run(predict_op, feed_dict={X: tsx[test_indices]})))




os.system('tensorboard  --logdir=' + ag.logdir)
# call('tensorboard', '--logdir='+ag.logdir)

