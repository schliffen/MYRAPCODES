#
# MY MODEL ON GPU
#
#
# lstm code 02
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
import tensorflow as tf
from keras import applications
import os.path as osp
from skimage import measure            # to find shape contour
# import scipy.ndimage as ndi            # to determine shape centrality
from skimage.feature import corner_harris, corner_subpix, corner_peaks, CENSURE

from six.moves import cPickle as pickle
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
# from keras_applications.mobilenet import MobileNet
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Input, Reshape
from keras.models import model_from_json
import coremltools
from FeatureCalculator import Features
#
print(keras.__version__)
#


from keras import backend as K
# import os
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
ap.add_argument('-d0','--data', type=str, default='/home/ali/BASKETBALL_DATA/in-out/DP1_v2/', help='direction to data dir')
ap.add_argument('-v','--sdir', type=str, default='/home/ali/BASKETBALL_DATA/in-out/', help='direction to save data')
ap.add_argument('-s','--generate_data', default=False,   help='wether to generate data from source or not')
ap.add_argument('--checkpoint', default='',  help='Path to the model checkpoint')
# ap.add_argument('--stopwords_file', default='/home/ali/CLionProjects/p_01/report/stopwords.txt', help='Path to stopwords file')
ap.add_argument('--summaries_dir', default='', help='Path to stopwords file')
ap.add_argument('-m','--model', default='MobileNet', nargs="?",
                # type=named_model,
                help='Name of the pre-trained model to use')
ag = ap.parse_args()

# Getting the information of available GPUs
#
# Configurations here
#
epochs = 30
batch_size = 16
#
# Loading Stored Data ----
#
# TODO: using dynamic model for length variable videos
# =============================

# loding pretrained model
# monet = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', include_top=True, pooling='avg')
#
# Reading the data ---
#
data_lst_in  = []
data_lst_out = []

def generate_data():
    for root, videos, _ in os.walk(ag.data):
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


with open(ag.sdir + 'DP1_in_out_shot_data_in1.pickle', 'rb') as handle:
    data_in = pickle.load(handle)
with open(ag.sdir + 'DP1_in_out_shot_data_out1.pickle', 'rb') as handle:
    data_out = pickle.load(handle)

#
# reading shots and feature extraction process
#
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Create some random colors
color = np.random.randint(0,255,(100,3))
#
fgbg = cv2.createBackgroundSubtractorMOG2()

shot1 =  []

for shots in data_in:
    for shot in list(shots):
        # try:
        #     img0 = cv2.imread(shot[0][-1])
        #     fgbg.apply(img0)
        # except:
        #     continue

        # tracker here

        # img_mean = img0.copy().astype(float)
        # weighted_diff_1 = np.zeros((img0.shape[0], img0.shape[1])).astype(float)
        # weighted_diff_2 = np.zeros((img0.shape[0], img0.shape[1])).astype(float)
        # weighted_diff_3 = np.zeros((img0.shape[0], img0.shape[1])).astype(float)
        # cul_bgsb = np.zeros_like(img0).astype(float)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        # hsv = np.zeros_like(img0)
        # hsv[...,1] = 255
        # bgr = np.zeros_like(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)).astype(float)
        frame_list = np.sort(list(shots[shot].keys()))
        for idx in frame_list:
            try:
                shot1.append( cv2.imread(shots[shot][idx][-1]) )
            except:
                continue

        shot1 = np.array(shot1)
        sum_diff = np.zeros_like(shot1[0]).astype(float)
        shot_mean = shot1.mean()
        shot_std = shot1.std()
        cul_bgsb = np.zeros_like(shot1[0])[:,:,0].astype(float)

        # background subtraction
        fgbg.apply(shot1[0])

        shot_magn1 = []
        shot_magn2 = []
        shot_magn3 = []

        for i in range(1, shot1.shape[0]):
                # gb0 = cv2.GaussianBlur(shot1[i-1],(5,5),0)
                # gb1 = cv2.GaussianBlur(shot1[i],(5,5),0)
                df1  = cv2.absdiff(cv2.GaussianBlur(shot1[i],(5,5),0), cv2.GaussianBlur(shot1[i-1],(5,5),0) )
                # pr_img = cv2.threshold(df1, 10, 255, 0, dst=cv2.THRESH_OTSU)[1].astype(float)
                bgr  =  cv2.erode(df1, kernel=(3,3), iterations =2)
                # fr_diff  = cv2.absdiff(shot1[i] , shot1[i-1])
                # pr_img = cv2.threshold(img_mot, 30, 255, 0, dst=cv2.THRESH_OTSU)[1].astype(float)
                # bgr  =  cv2.erode(pr_img, kernel=(3,3), iterations =1)

                # ----------------------------------------------------------------- summing up previous framee differences ------
                # sum_diff += ( i/(2*40*40) ) * bgr.astype(float)
                # sum_diff = cv2.erode(sum_diff, kernel=(5,5), iterations =7)
                # sum_diff = cv2.threshold(sum_diff, 20, 255, 0, dst=cv2.THRESH_OTSU)[1].astype(float)
                # cul_bgsb += cv2.absdiff(( idx/ shot1.shape[0] ) * bgr.astype(float), shot_mean)/shot_std
                # normalization
                # normal_bg = cv2.absdiff(cul_bgsb, shot_mean)/shot_std
                # plt.figure(1)
                # plt.imshow(fr_diff)
                # plt.figure(2)
                # plt.imshow(cul_bgsb)
                # plt.figure(3)
                # plt.imshow(sum_diff)
                # plt.figure(4)
                # plt.imshow(bgr)
                # ---------------------------------------------------------------------------------


                # features
                # img_building = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB
                # p0 = cv2.goodFeaturesToTrack(img0, mask = None) # , **feature_params
                # ------------------------------------------- (motion calculation) -----------------------------------------
                # motion vector calculation
                # p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, None, None, **lk_params)
                # prvs = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
                # next = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                # flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                # hsv[...,0] = ang*180/np.pi/2
                # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # img_mot = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # pr_img = cv2.threshold(img_mot, 30, 255, 0, dst=cv2.THRESH_OTSU)[1].astype(float)
                # bgr  =  cv2.erode(pr_img, kernel=(3,3), iterations =1)
                # bgr = pr_img # ( idx/ len(list(shot.keys())) ) *
                # draw the tracks
                # -----------------------------------------------------------------------------------------------

                # plt.figure(2)
                # plt.imshow(bgr)

                # frame_diff = cv2.absdiff(img_building, img0)
                #
                # cul_bgsb += ( idx/ len(list(shot.keys())) ) * frame_diff.astype(float)
                #
                #
                # plt.figure(1)
                # plt.imshow(frame_diff)
                # plt.figure(2)
                # plt.imshow(cul_bgsb)
                #
                # lets use background subtraction instead! -------------- this works somehow ------------------------------------
                #
                # fgmask1 = fgbg.apply(shot1[i])
                # plt.figure(1)
                # plt.imshow(fgmask1)
                #
                # fgmask2 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel).astype(float)
                # plt.figure(2)
                # plt.imshow(fgmask2)
                #
                # cul_bgsb += ( i/(2*40*40) ) * fgmask2
                #
                # plt.figure(3)
                # plt.imshow(cul_bgsb.astype('uint8'))
                # cv2.imwrite('/home/ali/CLionProjects/in_out_detection/Documentation_Reporting/frame_' + str(i) + '.jpg', shot1[i])
                # =======================================================

                # _, diff_thress_1 = cv2.threshold(frame_diff[:,:,0], 90, frame_diff.max(), 0, dst=cv2.THRESH_OTSU)
                # _, diff_thress_2 = cv2.threshold(frame_diff[:,:,1], 90, frame_diff.max(), 0, dst=cv2.THRESH_OTSU)
                # _, diff_thress_3 = cv2.threshold(frame_diff[:,:,2], 90, frame_diff.max(), 0, dst=cv2.THRESH_OTSU)

                # weighted_diff_1 += ( idx/len(list(shot.keys())) ) * diff_thress_1
                # weighted_diff_2 += ( idx/len(list(shot.keys())) ) * diff_thress_2
                # weighted_diff_3 += ( idx/len(list(shot.keys())) ) * diff_thress_3

                # plt.figure(1)
                # plt.imshow(frame_diff)

                # computing contours of the differentiated miage

                # imgray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                # ret, thresh = cv2.threshold(imgray, 100, imgray.max(),0, dst=cv2.THRESH_OTSU)
                # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #
                # # drawing imgs
                # img2 = shot1[i].copy()
                # for cit in range(np.min([len(contours),50])):
                #     cnt = contours[cit]
                #     cv2.drawContours(img2, [cnt], 0, (0,255,0), 1)
                #     # plt.figure(cit+10)
                #     # plt.imshow(img2)
                #
                # plt.figure(2) # , figsize=(56, 76)
                # plt.imshow(img2)
                # img3 = shot1[i].copy()
                # sift = cv2.xfeatures2d.SIFT_create()
                # kpi, desi = sift.detectAndCompute(img3, None)
                # img_building_keypoints = cv2.drawKeypoints(img3,
                #                                            kpi,
                #                                            img3,
                #                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
                # plt.figure(3)
                # plt.title('SIFT Interest Points')
                # plt.imshow(img_building_keypoints)

                img4 = next = cv2.cvtColor(shot1[i].copy(), cv2.COLOR_BGR2GRAY)
                surf = cv2.xfeatures2d.SURF_create()
                kpu, desu = surf.detectAndCompute(img4, None)
                # img_building_keypoints = cv2.drawKeypoints(img4,
                #                                            kpu,
                #                                            img4,
                #                                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
                # plt.figure(4)
                # plt.title('SURF Interest Points')
                # plt.imshow(img_building_keypoints) #; plt.show()

                # another feature points
                # ---------------------------------------------------------
                # # img5 = shot1[i].copy()
                # img5 = cv2.cvtColor(shot1[i].copy(), cv2.COLOR_BGR2GRAY)
                # detector = CENSURE()
                # detector.detect(img5)
                # coords = corner_peaks(corner_harris(img5), min_distance=3)
                # coords_subpix = corner_subpix(img5, coords, window_size=13)
                # ------------------------------------------------------------
                # plt.figure(5)
                # plt.title(' Interest Points')
                # plt.imshow(img_building_keypoints); plt.show()

                # img0 = img1
                point_list = []
                mag1 = []
                # mag2 = []
                # mag3 = []
                #
                for i in range(min(len(kpu), 10)):
                    # point_list.append(kpu[i].pt)
                    mag1.append( img4[int(kpu[i].pt[1]), int(kpu[i].pt[0]), 0])
                    np.vstack((mag1, img4[int(kpu[i].pt[1]), int(kpu[i].pt[0]), 0]))
                    # mag2.append( img4[int(kpu[i].pt[1]), int(kpu[i].pt[0]), 1])
                    # mag3.append( img4[int(kpu[i].pt[1]), int(kpu[i].pt[0]), 2])
                #
                shot_magn1.append(np.array(mag1))
                # shot_magn2.append(np.array(mag2))
                # shot_magn3.append(np.array(mag3))

        # computing shot features here
        feature_calculator = Features()
        shot_magn1 = np.array(shot_magn1).transpose(1,0)
        # shot_magn2 = np.array(shot_magn2).transpose(1,0)
        # shot_magn3 = np.array(shot_magn3).transpose(1,0)

        result = feature_calculator.meanvariance(shot_magn1[0])




                    # computing the magnitudes





                # computing statistics of the sequenced frames





    # temporal feature extraction process
        



"""
MHI_DURATION = 10
DEFAULT_THRESHOLD = 32
def func2():
    import sys
    try: video_src = sys.argv[1]
    except: video_src = './images/example-%03d.jpg'

    cv2.namedWindow('motion-history')
    cv2.namedWindow('raw')
    cv2.moveWindow('raw', 200, 0)
    while True:
        cam = cv2.VideoCapture(video_src)
        ret, frame = cam.read()
        h, w = frame.shape[:2]
        prev_frame = frame.copy()
        motion_history = np.zeros((h, w), np.float32)
        timestamp = 0
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame_diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
            timestamp += 1

            # update motion history
            cv2.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)

            # normalize motion history
            mh = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
            cv2.imshow('motempl', mh)
            cv2.imshow('raw', frame)

            prev_frame = frame.copy()
            if 0xFF & cv2.waitKey(5) == 27:
                break
    cv2.destroyAllWindows()


# --------------------- Ozal Codes ----------------------
# history = model.fit(TrX, trY,  validation_split=.2,  batch_size=24, nb_epoch=epochs)
# loss, acc = model.evaluate(TsX, tsY)
# print('model evaluation; accuracy:', acc , 'loss', loss)
#
#
# print('model evaluation; accuracy:', acc , 'loss', loss)

# saving the results of the model

# checking the accuracy
# loss, acc = model.evaluate(TsX, tsY)
# print('model evaluation; accuracy:', acc , 'loss', loss)
# checking visual examplr
# rnd = np.random.randint(0,tsY.shape[0]-20,1)
# print('model prediction for some examples: ')
# model.predict_classes(TsX[rnd:rnd+20])

# print('ground truth for those examples: ')
# print(tsY[rnd:rnd+20])

# # serialize model to JSON
# model_json = model.to_json()
# with open("model_256.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_256.h5")
# print("Saved model to disk")

# saving coremltool ...

# coreml_model = coremltools.converters.keras.convert(model)
# coreml_model.save('model_256_test_01.mlmodel')
#
"""