#
# data augmentation with data generator
#
import os,sys, glob
import numpy as np
import random
import cv2
import _pickle as pickle
from keras.preprocessing.image import ImageDataGenerator
# image augmentator
import imgaug as ia
from imgaug import augmenters as iaa
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import argparse
args = argparse.ArgumentParser()
args.add_argument('-d1', '--input_path', default='/home/ali/BASKETBALL_DATA/DEBUG/test/', help='path to input data')
args.add_argument('-d2','--output_path', default='/home/ali/BASKETBALL_DATA/DEBUG/augmented_test/', help='path to output data')


arg = args.parse_args()

count = 10

#
sometimes = lambda aug: iaa.Sometimes(0.9, aug)
seq = iaa.Sequential(
    [
        # apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        # iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -5% to 10% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.01, 0.01),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10), # rotate by -45 to +45 degrees
            # shear=(-10, 10), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 0.1), n_segments=(2, 10))
                       # ), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                           iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
                           iaa.MedianBlur(k=(1, 3)), # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 0.4), lightness=(0.3, .9)), # sharpen images
                       # iaa.Emboss(alpha=(0, 0.04), strength=(0, .08)), # emboss images
                       # search either for all edges or for directed edges,
                       # blend the result with the original image using a blobby mask
                       iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.25, .5)),
                           iaa.DirectedEdgeDetect(alpha=(0.1, 0.4), direction=(0.0, 1.0)),
                       ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.03), per_channel=0.2), # randomly remove up to 10% of the pixels
                           # iaa.CoarseDropout((0.03, 0.05), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       # iaa.Invert(0.05, per_channel=True), # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                       iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                       # either change the brightness of the whole image (sometimes
                       # per channel) or change the brightness of subareas
                       iaa.OneOf([
                           # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.FrequencyNoiseAlpha(
                               exponent=(-4, 0),
                               # first=iaa.Multiply((0.5, 1.5), per_channel=True),
                               second=iaa.ContrastNormalization((0.5, 2.0))
                           )
                       ]),
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                       iaa.Grayscale(alpha=(0.0, 1.0)),
                       # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                       # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)

# creating augmented image!
# in data

X = []
Y = []
n_augmentation = 5
for root, folders, _ in os.walk(arg.input_path):
    for folder in folders:
        img_list = glob.glob(root + folder + '/*.jpg')
        for item in img_list:
            img = cv2.imread(item)
            cv2.imwrite(arg.output_path + folder + '/'+ item.split('/')[-1] , img)
            for i in range(n_augmentation):
                images_aug = seq.augment_image(img)
                cv2.imwrite(arg.output_path + folder + '/'+  item.split('/')[-1].split('.')[0] +'_' + str(i) + '_aug.jpg', images_aug)

