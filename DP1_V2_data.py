#
# reading data from stored folders
#
import os, sys, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
#
import cv2
import argparse
import keras
import numpy as np

from skimage import measure            # to find shape contour
import scipy.ndimage as ndi            # to determine shape centrality
from skimage.feature import corner_harris, corner_subpix, corner_peaks, CENSURE

from six.moves import cPickle as pickle
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
# from keras_applications.mobilenet import MobileNet
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Input, Reshape
from keras.models import model_from_json
import coremltools
#


# def set_keras_backend(backend):
#
#     if K.backend() != backend:
#         os.environ['KERAS_BACKEND'] = backend
#         reload(K)
#         assert K.backend() == backend
# set_keras_backend('mxnet')

#
ap = argparse.ArgumentParser()
ap.add_argument('-d0','--data', type=str, default='/home/ali/BASKETBALL_DATA/DP1_v1_2/', help='direction to data dir')
ap.add_argument('-v','--sdir', type=str, default='/home/ali/BASKETBALL_DATA/in-out/', help='direction to save data')
ap.add_argument('-s','--generate_data', default=True,   help='wether to generate data from source or not')
ag = ap.parse_args()

# Getting the information of available GPUs
#
# Configurations here
#
epochs = 30
batch_size = 16


# Loading Stored Data ----


# Reading the data ---
#
data_lst_in  = []
data_lst_out = []
def generate_data():
    for root, videos, _ in os.walk(ag.data):
        #

        #
        for video in videos:
            #
            cls_lst = glob.glob(root + video + '/*')
            #
            shot_dict_in = {}
            shot_dict_out = {}
            #
            for classes in cls_lst:
                if classes.split('.')[-1] == 'txt':
                    continue
                frames = glob.glob(classes + '/*')

                # ------------------------------------ shot collection
                frame_dict_in = {}
                frame_dict_out = {}
                shot_list_in = []
                shot_list_out = []

                for frame in frames:
                    # extracting image details ----
                    id = frame.split('/')[-1].split('_')[0][2:]
                    sn = frame.split('/')[-1].split('_')[1][2:]
                    fn = frame.split('/')[-1].split('_')[2][3:]
                    lbl = classes.split('/')[-1]
                    # updating the dictionaries
                    if lbl == 'in':
                        try:
                            if sn in shot_list_in:
                                shot_dict_in[sn].update( {int(fn): [id, frame] })
                            else:
                                shot_list_in.append(sn)
                                shot_dict_in.update({sn: {int(fn): [id, frame] }})

                        except:
                            id = frame.split('/')[-1].split('_')[1][:]
                            sn = frame.split('/')[-1].split('_')[2][2:]
                            fn = frame.split('/')[-1].split('_')[3][3:]
                            if sn in shot_list_in:
                                shot_dict_in[sn].update( {int(fn): [id, frame] })
                            else:
                                shot_list_in.append(sn)
                                shot_dict_in.update({sn: {int(fn): [id, frame] }})

                    elif lbl == 'out':
                        try:
                            if sn in shot_list_out:
                                shot_dict_out[sn].update( {int(fn): [id, frame] })
                            else:
                                shot_list_out.append(sn)
                                shot_dict_out.update({sn: {int(fn): [id, frame] }})
                        except:
                            id = frame.split('/')[-1].split('_')[1][:]
                            sn = frame.split('/')[-1].split('_')[2][2:]
                            fn = frame.split('/')[-1].split('_')[3][3:]
                            if sn in shot_list_out:
                                shot_dict_out[sn].update( {int(fn): [id, frame] })
                            else:
                                shot_list_out.append(sn)
                                shot_dict_out.update({sn: {int(fn): [id, frame] }})

                # --------------------------------
                if classes.split('/')[-1] == 'in':
                    data_lst_in.append(shot_dict_in)
                elif classes.split('/')[-1] == 'out':
                    data_lst_out.append(shot_dict_out)

    with open(ag.sdir + 'DP1_in_out_shot_data_in1.pickle', 'wb') as f:
        pickle.dump(data_lst_in, f)
    with open(ag.sdir + 'DP1_in_out_shot_data_out1.pickle', 'wb') as f:
        pickle.dump(data_lst_out, f)

    print('data is prepared!')

if ag.generate_data:
    generate_data()


with open(ag.sdir + 'DP1_in_out_shot_data_in.pickle', 'rb') as handle:
    data_in = pickle.load(handle)
with open(ag.sdir + 'DP1_in_out_shot_data_out.pickle', 'rb') as handle:
    data_out = pickle.load(handle)
#
