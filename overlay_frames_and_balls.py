#
# Preprocessing the data in order to figure out the feeding features
#
# imports
import os,sys,glob
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
#
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--ddir', type=str, default='/home/yizhuo/Documents/deep_learning/data_/new_labelled/',
                help='the data dir')


arg = ap.parse_args()
# --------------------------



# collecting the data list
data_list = os.listdir(arg.ddir)


# analysing the data for extracting proper features
img =[]
kernel_size = (15,15)

for i, frame in enumerate(data_list):
    if img == []:
        img = cv2.imread(arg.ddir + frame)
        bimg = cv2.GaussianBlur(img, kernel_size,0)
        bgimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=100, detectShadows=False)
        fgmask = fgbg.apply(bgimg)
        ave_frames = np.zeros(bgimg.shape)
        continue

    image = cv2.imread(arg.ddir + frame)
    # some preparation with filters
    bimage = cv2.GaussianBlur(image, kernel_size,0)
    bgimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # background subtraction
    fgmask = fgbg.apply(bimage)
    # TODO: take this as mask in order to extract exact ball and put them together
    plt.figure('mog background subtraction- ' + str(i)); plt.imshow(fgmask)
    #cv2.imshow('fgmask',frame)
    #cv2.imshow('frame',fgmask)
    gsim = cv2.absdiff(bgimage, bgimg)
    # plt.figure('gray- ' + str(i)); plt.imshow(gsim)
    csim = cv2.absdiff(bimage,bimg)
    plt.figure('color- '+ str(i)); plt.imshow(csim)

    # thresh = cv2.threshold(gsim, 35, 255, cv2.THRESH_BINARY)[1]


    ave_frames += 1./(len(data_list)-1) * fgmask
    print(frame)
    plt.figure('average- ' + str(i)); plt.imshow(ave_frames); plt.show()


    #TODO: use this idea in order to increase the accuracy of the goal recognition.

    #TODO: make decision based on the corruptions of the following image or version of that
plt.figure('vaerage ball motion '); plt.imshow(ave_frames)
plt.figure('fusion-image '); plt.imshow(2.*ave_frames + bgimage); plt.show()



    # Analysing with flow motion of the ball
    # img_flow = cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    # flow = cv2.calcOpticalFlowFarneback(bgimg, bgimage, bgimg, 0.5,1,3,15,3,5,1)
    # bgimg = bgimage
    # plt.figure('flow motion- ' + str(i)); plt.imshow(flow[:,:,0] + flow[:,:,1]); plt.show()


    # cv2.imshow(sim)
    # cv2.waitKey(0)
    # if cv2.waitKey(0) & 0xFF == ord("q"):
    # #    continue
    #     break
    #     cv2.destroyAllWindows()











