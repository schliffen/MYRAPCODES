#
# script for loading and using data
# ALI NEHRANI for RAPSODO
# TODO: to be organized
import os,glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from six.moves import cPickle as pickle
import cv2
import h5py
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from MY_LSTM_feat_ext import feature_extraction_n as fext
import time
class sample_data():
    def __init__(self):

        # self.type == None
        #define constants
        #unrolled through 28 time steps
        self.time_steps=28
        #hidden LSTM units
        self.num_units=128
        #rows of 28 pixels
        self.n_input=28
        #learning rate for adam
        self.learning_rate=0.001
        #mnist is meant to be classified in 10 classes(0-9).
        self.n_classes=10
        #size of batch
        self.batch_size=128
    #import mnist dataset
    def mnist_(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
        return mnist

    def k_digits(self):
        digits = load_digits()
        trX, tsX, trY, tsY = train_test_split(digits.data, digits.target)
        # setting the test size
        trx = trX[:-7]; tryy = trY[:-7];
        trTg = np.array([item for num,item in enumerate(tryy) if num % self.time_steps == 0])
        tsTg = np.array([item for num,item in enumerate(tsY) if num % self.time_steps == 0])
        trx = trx.reshape(-1, self.time_steps, 64)
        tsx = tsX.reshape(-1, self.time_steps, 64)
        test_size = tsTg.shape[0]
        # one_hot encoding of the target
        enc = OneHotEncoder()
        arenc = enc.fit(digits.target.reshape(-1,1))
        trY = arenc.transform(trTg.reshape(-1,1)).toarray()
        tsY = arenc.transform(tsTg.reshape(-1,1)).toarray()


    def video_data(self, vidir):
        # try:
        # TODO: check wether the data exists
        #
        features = False
        data_path = 'pre_feature_data.pickle' if features else 'pre_data.pickle'
        if not os.path.isfile(vidir + data_path):

            if features:
                lfext = fext('MobileNet'); # this is to creat mobilenet features
            data = []
            for root, folders,_ in os.walk(vidir):

                for file in folders:
                    sub_data = []
                    frame_lst = glob.glob(root  + file + '/*.png')
                    try:
                        label_lst = [glob.glob(root + file + '/*.txt')[i] for i in range(2)
                                 if glob.glob(root + file + '/*.txt')[i].split('/')[-1].split('.')[0] == file
                                 ][0]
                    except:
                        print(file)
                    with open(label_lst) as lf:
                        content = lf.readlines()
                        sub_data = [{-1 : x.strip()} for x in content[1:]]
                    # iterating all available frames on the video
                    for frame in frame_lst:
                        sh_fr = frame.split('/')[-1].split('_')[-3:-1]
                        img = cv2.imread(frame)
                        img = np.resize(img, (128,128,3)) # .flatten() # do not need flatten for feature
                        # computing mobilenet features and storing them

                        if features:
                              # decide wether to save features
                            start = time.time()
                            img_features = lfext.get_feature(img)
                            print('elapsed time: ', time.time() - start)
                        else: img_features = img.flatten();
                        try:
                            if content[1][0] == '0':
                                sub_data[int(sh_fr[0])].update({int(sh_fr[1]): img_features})
                            elif content[1][0] == '1':
                                sub_data[int(sh_fr[0])-1].update({int(sh_fr[1]): img_features})
                            else:
                                print('list naming started from' + content[1][0])
                        except:
                            print('an error is accured!')
                        print('frame %s is added to the data' % frame)
                    # adding this video to other videos
                    data.append(sub_data)
            if not features:
                with open(vidir + 'pre_data.pickle', 'wb') as f:
                    pickle.dump(data,f)
                print('pre feature data is saved! with success')
            else:
                with open(vidir + 'pre_feature_data.pickle', 'wb') as f:
                    pickle.dump(data,f)
                print('pre data is saved! with success')
            return [],[]
        else:
            if features:
                with open(vidir + 'pre_feature_data.pickle', 'rb') as f:
                    prdata = pickle.load(f)
                print('pre feature data is loaded ....')
            else:
                with open(vidir + 'pre_data.pickle', 'rb') as f:
                    prdata = pickle.load(f)
                print('pre data is loaded with success ....')
            vid_ba_01_data  = []
            vid_ba_01_label = []
            for i in range(len(prdata)):

                for j in range(len(prdata[i])):
                    sh_array = []
                    d_items = sorted(prdata[i][j].items(), key= lambda x:x[0])
                    label = d_items[0][1]
                    for item in d_items[1:]:
                        sh_array.append(item[1])

                    try:
                        if not int(label.split(' ')[1]) == -1:
                            vid_ba_01_data.append(sh_array)
                            vid_ba_01_label.append(int(label.split(' ')[1]))
                        # del sh_array
                    except:
                        print(item)

            if features:
                with open(vidir + 'video_ba_feat_01_data.pickle', 'wb') as f:
                        pickle.dump(np.array(vid_ba_01_data), f)
                print('post feature data is saved with success ....')
                with open(vidir + 'video_ba_feat_01_label.pickle', 'wb') as f:
                        pickle.dump(np.array(vid_ba_01_label), f)
                print('post feature label is saved with success ....')

            else:
                with open(vidir + 'video_ba_02_data.pickle', 'wb') as f:
                        pickle.dump(np.array(vid_ba_01_data), f)
                print('post data is saved with success! ....')
                with open(vidir + 'video_ba_02_label.pickle', 'wb') as f:
                        pickle.dump(np.array(vid_ba_01_label), f)
                print('post label is saved with success! ....')



        # except Exception as ex:
        #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #     message = template.format(type(ex).__name__, ex.args)
        #     print(message)
        #     print(file)
        # all data is in a list
        # puting data in a single multi array in a sorted form as numpay array

        return vid_ba_01_data, vid_ba_01_label



