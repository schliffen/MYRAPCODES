#
# Trying to underestand whether ball entered in the frame or not
#  Using ball confirmation alg to check it out
#
import numpy as np
import cv2
#
# Preprocessing the data in order to figure out the feeding features
#
# imports
import os,sys,glob
import numpy as np
from numpy import linalg as LA
import cv2
import argparse
import matplotlib.pyplot as plt
#
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--ddir', type=str, default='/home/ali/BASKETBALL_DATA/aa sorted by shot/1/',
                help='the data dir')
ap.add_argument('-t', '--tnm', type=str, default='net_rect.txt',help='the text file for hoop coordinates')


arg = ap.parse_args()
# --------------------------

# collecting the data list
data_list = os.listdir(arg.ddir)


# analysing the data for extracting proper features
img =[]
kernel_size = (15,15)


# reading the hoop coordinates
with open(arg.ddir + arg.tnm, 'r') as f:
    coord = f.read().split(' ')

data_list = '/home/ali/BASKETBALL_DATA/test/'
cropped_videos = '/home/ali/BASKETBALL_DATA/cropped_2/'
imgs = sorted(os.listdir(data_list))

croped = []
for i, frame in enumerate(imgs):
    if croped == []:
        sfr = cv2.imread(data_list + frame)
        gfr = cv2.cvtColor(sfr, cv2.COLOR_BGR2GRAY)
        # the cropped scenario
        croped = gfr[int(np.round(float(coord[1])))-20:int(np.round(float(coord[1])))+ int(np.round(float(coord[3]))),
                 int(np.round(float(coord[0]))):int(np.round(float(coord[0])))+int(np.round(float(coord[2])))]
        fgbg = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=100, detectShadows=False)
        fgmask1 = fgbg.apply(croped)
        # ave_frames = np.zeros(bgimg.shape)
        continue
    # plt.imshow(fgmask1)
    sfr = cv2.imread(data_list + frame)
    gfr = cv2.cvtColor(sfr, cv2.COLOR_BGR2GRAY)
    # the cropped scenario
    croped_1 = gfr[int(np.round(float(coord[1])))-20:int(np.round(float(coord[1])))+ int(np.round(float(coord[3]))),
             int(np.round(float(coord[0]))):int(np.round(float(coord[0])))+int(np.round(float(coord[2])))]
    bimage = cv2.GaussianBlur(croped_1, kernel_size,0)
    # bgimage = cv2.cvtColor(bimage, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg.apply(bimage)
    # checking the countures and keeping big changes as video frame
    edged = cv2.Canny(fgmask2, 30, 200)
    # erosion and dilation
    kernel = np.ones((3,5), np.uint8)
    plt.imshow(fgmask2)
    img_erosion = cv2.erode(fgmask2, kernel, iterations=15)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=5)
    image, contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:2]
    screenCnt = None
    plt.imshow(fgmask2)
    for c in cnts:
        # approximate the contour
        area = cv2.contourArea(c)
        # hull = cv2.convexHull(cnt)
        print(area)
        print(frame)
        # filtering by area
        if area > 30:
            # np.save(cropped_videos + 'cropped_' + frame, croped_1)
            cv2.imwrite(cropped_videos + 'cropped_' + frame, croped_1)
            # with open(cropped_videos + 'cropped_' + frame, 'wb') as f:
            #     f.write()

            try:
                cv2.drawContours(croped_1, [c], -1, (0, 255, 0), 3)
                cv2.imshow("Game Boy Screen", croped_1)
                cv2.waitKey(0)
            except: continue




    plt.Figure()
    plt.imshow(fgmask2)
    plt.Figure()
    plt.imshow(croped_1)
    # difference = LA.norm(croped - croped_1)
    # croped = croped_1
    # print('the difference', difference, 'frame', i)



#     image = cv2.imread(arg.ddir + frame)
#     # some preparation with filters
#     bimage = cv2.GaussianBlur(image, kernel_size,0)
#     bgimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#
#     # background subtraction
#     fgmask = fgbg.apply(bimage)
#     # TODO: take this as mask in order to extract exact ball and put them together
#     plt.figure('mog background subtraction- ' + str(i)); plt.imshow(fgmask)
#     #cv2.imshow('fgmask',frame)
#     #cv2.imshow('frame',fgmask)
#     gsim = cv2.absdiff(bgimage, bgimg)
#     # plt.figure('gray- ' + str(i)); plt.imshow(gsim)
#     csim = cv2.absdiff(bimage,bimg)
#     plt.figure('color- '+ str(i)); plt.imshow(csim)
#
#     # thresh = cv2.threshold(gsim, 35, 255, cv2.THRESH_BINARY)[1]
#
#
#     ave_frames += 1./(len(data_list)-1) * fgmask
#     print(frame)
#     plt.figure('average- ' + str(i)); plt.imshow(ave_frames); plt.show()
#
#
#     #TODO: use this idea in order to increase the accuracy of the goal recognition.
#
#     #TODO: make decision based on the corruptions of the following image or version of that
# plt.figure('vaerage ball motion '); plt.imshow(ave_frames)
# plt.figure('fusion-image '); plt.imshow(2.*ave_frames + bgimage); plt.show()



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











