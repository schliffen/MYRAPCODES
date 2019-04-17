#
#
#
import sys,os, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/data_01/'
save_dir = '/home/ali/BASKETBALL_DATA/BBcropped/'
# in_data = os.listdir(data_dir)
# out_data = os.listdir(data_dir + 'out')

for _,folders,_ in os.walk(data_dir):

    for folder in folders:
        img_list = glob.glob(data_dir + folder + '/*.png' )
        for item in img_list:
            img = cv2.imread(item)
            # nimg = img[img.shape[0]//2:,:]
            img = cv2.resize(img, (130,130))
            plt.imshow(img)
            cv2.imwrite(save_dir + str(item.split('/')[-1].split('.')[0]) + '.jpg', img)


