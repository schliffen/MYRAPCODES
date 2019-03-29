#
# This Script is designed for checking the images and selecting data for retrain process
#
""" this script is for collecting data,
 all controls should be done by eyes

"""

import sys,os, glob
import numpy as np
import cv2

from src.BackBoard_Detection_Py import BackBoard_Detection
from src.Net_Detection_Py import Net_Detection
from src.Visualization_BndBox import Visualize

import matplotlib.pyplot as plt


import argparse
args = argparse.ArgumentParser()
args.add_argument('-d0', '--inp_data', default='/home/rapsodo/BASKETBALL/00_DATASETS/Ali_Hoca_BB_Test_Images/', help= 'the input data directory')
args.add_argument('-d1', '--out_data', default='/home/rapsodo/BASKETBALL/00_DATASETS/BB_NET_detection_data/', help= 'the collected data directory')
args.add_argument('-d2', '--BB_fg_model', default='/home/rapsodo/CLionProjects/rapsodo_ball_detection/FrozenGraph/backboard_detection_2.pb', help= 'the backboard detection model')
args.add_argument('-d3', '--NT_fg_model', default='/home/rapsodo/CLionProjects/rapsodo_ball_detection/FrozenGraph/net_detector_v1_04.pb', help= 'the hoop detection model')


ap = args.parse_args()


# introducing backboard detector
BB_Det = BackBoard_Detection(ap.BB_fg_model)

# introducing hoop detector
Net_Det = Net_Detection(ap.NT_fg_model)

# visualization


#

data_list = glob.glob(ap.inp_data + '*.jpg')

data_list.append(glob.glob(ap.inp_data + '*.JPG'))

refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False

# setting cropper here!
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

for counter, frame in enumerate(data_list):
    try:
        print('number of seen images: ', counter)
        write_bb = False
        write_net = False
        #

        img = cv2.imread(frame)

        img = cv2.resize(img, (512,512))

        bb_rect = BB_Det.get_BackBoard_BoundingBoxes(img)


        # visualizing the results
        frame = img.copy()

        # visualizing backboard
        Visualize.visualize(frame, bb_rect[0])



        #
        cv2.imshow('For Backboard: In case of wrong result, enter -s- to save', frame)
        if cv2.waitKey(0) == ord('s'):
            #
            write_bb = True



        else: # backboard detector is working well


            # visualizing the hoop
            h = bb_rect[0][3] - bb_rect[0][1]
            w = bb_rect[0][2] - bb_rect[0][0]
            crp_img = img[bb_rect[0][1] :bb_rect[0][3] ,
                      bb_rect[0][0] :bb_rect[0][2] ]

            # Looking for hoop in the backboard image
            net_rect = Net_Det.get_Net_BoundingBoxes(crp_img)
            # visualizing net

            print('Detection score: ', net_rect[1])

            crp_img_vis = crp_img.copy()
            Visualize.visualize(crp_img_vis, net_rect[0])

            #
            cv2.imshow('For Net: enter -s- to save', crp_img_vis)
            if cv2.waitKey(0) == ord('s'):
                #
                write_net = True


        # check wether to write backboard
        if write_bb == True:
            # random number generation for different naming
            rnd_num = np.random.randint(0, len(data_list), 1)[0]

            cv2.imwrite(ap.out_data + 'backboard_data' + '_' + str(rnd_num) + '_' + str(counter) + '.jpg', img)

            # looking for hoop on wrongly detected backboard by getting two points as an input

            image = img.copy()
            cv2.imshow("image", image)
            # key = cv2.waitKey(1) & 0xFF
            cv2.waitKey(0)

            if len(refPt) == 2:

                h = refPt[1][0] - refPt[0][0]
                w = refPt[1][1] - refPt[0][1]

                roi = image[refPt[0][1]-int(w/15.):refPt[1][1]+int(w/15.), refPt[0][0]-int(h/15.):refPt[1][0]+int(h/15.)]
                cv2.imshow("ROI", roi)
                if cv2.waitKey(0) == ord('s'):
                    # random number generation for different naming
                    rnd_num = np.random.randint(0, len(data_list), 1)[0]

                    cv2.imwrite(ap.out_data + 'hoop_data' + '_' + str(rnd_num) + '_' + str(counter) + '.jpg',roi)

        # check wether to write hoop img
        if write_net == True:
            # random number generation for different naming
            rnd_num = np.random.randint(0, len(data_list), 1)[0]

            h = bb_rect[0][3] - bb_rect[0][1]
            w = bb_rect[0][2] - bb_rect[0][0]
            cv2.imwrite(ap.out_data + 'hoop_data' + '_' + str(rnd_num) + '_' + str(counter) + '.jpg',
                    img[bb_rect[0][1]-int(h/15.):bb_rect[0][3]+int(h/15.), bb_rect[0][0]-int(w/15.):bb_rect[0][2]+int(w/15.)])

    except:
        continue

# close all open windows
cv2.destroyAllWindows()