#
# Preparing the dataset for training
#
# imports
import os, sys, glob
import numpy as np
from PIL import Image
#from array improt *
from random import shuffle
import argparse
import h5py
#
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--imgdir", type=str, help="the direction of the images", default='/home/yizhuo/Documents/deep_learning/data_/labelled_renamed/')
ap.add_argument("-r","--ratio", type=float, help='ration of train and test', default=.27)
arg = ap.parse_args()


"""
## LOADING THE DATA ----------
gold_set = open("gold_list.txt", "w")
gold_set.write("List of the raw image data \n")
gold_set.close()
# iterating over the folders ----
for folders in os.listdir(arg.imgdir):
    for _, sfolders, _ in os.walk(arg.imgdir + folders):
        for ssfolders in sfolders:
            if ssfolders == []: continue
            for _, _, imgs in os.walk(arg.imgdir + folders + '/' + ssfolders):
                with open('gold_list.txt', "a") as gold_set:
                    for item in imgs:
                        gold_set.write(str(arg.imgdir + folders + '/' + ssfolders + '/' + item) + '\n')
#
"""



# ---------------------------------------
# part 1: preprocessing the data
# ----------------------------------------










# -------------------------------------------
# PART 2: saving data into standard format
# -------------------------------------------
#

Names = ['train', 'validation', 'test']

with open('gold_list.txt','r') as idir:
    content = idir.readlines()
    data_list = [item.strip() for item in content]
data_list = data_list[1:]

# shuffling the data list to divide into train and test images










#
#train_addrs = addrs[0:int(0.6*len(addrs))]
#train_labels = labels[0:int(0.6*len(labels))]
#val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
#val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
#test_addrs = addrs[int(0.8*len(addrs)):]
#test_labels = labels[int(0.8*len(labels)):]

