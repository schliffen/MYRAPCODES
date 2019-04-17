#
import cv2
import os
import os.path
import numpy as np
import mxnet.image as image
import numexpr as ne
import tables as tb
from tables import *
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

#DATA_DIR = '/tmp/data'
MAIN_DIR = ''
DATA_DIR = 'labledplt/'
IMG_DIR = '15_22/'
Ext_data = 'data/'

#path_to_images = '/home/bayes/Academic/Research/Radarsan-01/ANPR/12_19/'
#label_dict = {'0':0,  '1':1, '2':2,  '3':3,  '4':4,  '5':5,  '6':6,  '7':7,  '8':8, '9':9,\
#              'a':10, 'b':11,'c':12, 'd':13, 'e':14, 'f':15, 'g':16, 'h':17, 'i':18,\
#              'j':19, 'k':20,'l':21, 'm':22, 'n':23, 'o':24, 'p':25, 'q':26, 'r':27,\
#              's':28, 't':29,'u':30, 'v':31, 'w':32, 'x':33, 'y':34,'z':35}
# reverse dictionaly for the case of loading data
#reverse_dict = dict(zip(label_set.values(), label_set.keys()))

# getting other type of data
desired_x = 1500
desired_y = 2000
train_iter = image.ImageDetIter(
        batch_size=300,
        data_shape=(3, 256, 256),
        path_imgrec='data/pikachu_train.rec',
        path_imgidx='data/pikachu_train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
for exd in train_iter:
    extimg = exd.data
    break

label_dict = {'plate':0}
# functions for adding noise
def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

   elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy
#
def padding(img,desired_x,desired_y):
    old_x = img.shape[0] # old_size is in (height, width) format
    old_y = img.shape[1]
    delta_w = desired_x - old_x
    delta_h = desired_y - old_y
    top = delta_w//2; bottom = delta_w - top
    left = delta_h//2; right = delta_h - left
    #color = [0, 0, 0] value=color
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return img, top, left
    #




# read the xml file
for _, _, files in os.walk(DATA_DIR):
    for file in files:
        try:
            tree = ET.parse(DATA_DIR + file)
            root = tree.getroot()
            #
            file_name = root.findall('filename')[0].text
#
            objects = root.findall('object')
            #img_path =  root.findall('path')[0].text
            #
            # creating label
            #
            label = []
            #
            for object in objects:
                sub_label = []
                dig_class = object.findall('name')[0].text
                # checking if the object is in the plate i.e. in the dictionary,
                if dig_class == 'plate':
                    sub_label = np.zeros(5)
                    sub_label[0] = label_dict[dig_class] # I am using the valuse of the class dictionary
                    elem = object.findall('bndbox')
                    sub_label[1] = eval(elem[0][0].text)
                    sub_label[2] = eval(elem[0][1].text)
                    sub_label[3] = eval(elem[0][2].text)
                    sub_label[4] = eval(elem[0][3].text)
                    label.append(np.int0(sub_label))
## ------------------------------------------
# loading corresponding images
#
            ImData = cv2.imread(IMG_DIR + file_name)
            #ImData = cv2.imread(img_path)
            try:
                ImData = cv2.cvtColor(ImData, cv2.COLOR_BGR2GRAY)
            except:
                print('This image itself is in gray scale')
#
# Preprocessing for problems resizing
#
            ImData, top, left = padding(ImData, desired_x, desired_y)
            # making more data with opencv
            # 3. putting plate in an other image
            w = np.zeros((len(label))); h = np.zeros((len(label)))
            rx = np.zeros((len(label))); ry = np.zeros((len(label)))
            xmin, xmax, ymin, ymax = np.zeros((len(label))), np.zeros((len(label))), np.zeros((len(label))), np.zeros((len(label)))

            for j in range(len(label)):
                xmin[j] = label[j][1] + left
                ymin[j] = label[j][2] + top
                xmax[j] = label[j][3] + left
                ymax[j] = label[j][4] + top
                w[j] = xmax[j] - xmin[j]
                h[j] = ymax[j] - ymin[j]
                rx[j] = np.random.randint(0, ImData.shape[0] - w[j], 1)
                ry[j] = np.random.randint(0, ImData.shape[1] - h[j], 1)


            # creating artifitial data images
            exlabel = []
            eximgs = []
            # select a random image
            for j in range(5):
                img_index = np.random.randint(0,300,1)
                temp_img = extimg[0][img_index,:,:,:].asnumpy()[0,:,:,:]
                #temp_img  = cv2.cvtColor(temp_img[], cv2.COLOR_BGR2GRAY)
                temp_img = temp_img[0,:,:]
                temp_img, _, _ = padding(temp_img, desired_x, desired_y)
                for i in range(len(label)):
                    exsublabel = np.zeros(5)
                    exsublabel[0] = label_dict[dig_class]
                    exsublabel[1],exsublabel[2] =  rx[i], ry[i]
                    exsublabel[3], exsublabel[4] = rx[i] + w[i], ry[i] + h[i]
                    temp_img[int(rx[i]):int(rx[i] + w[i]), int(ry[i]):int(ry[i] + h[i])] = ImData[int(ymin[i]):int(ymax[i]), int(xmin[i]):int(xmax[i])]
                    exlabel.append(exsublabel)
                    eximgs.append(temp_img)

            # 2. dropping out
            # to implement it later
            # 1. adding noise
            ImData_2 = noisy('s&p', ImData)

            # putting data together
            ImData   = ImData.reshape(1,-1)
            ImData_2 = ImData_2.reshape(1, -1)
            # post processing on the images
            im_size = ImData.shape[1]
            Sub_Data_Array = np.zeros((len(label), im_size + 5 + 2))
            Sub_Data_Array2 = np.zeros((len(label), im_size + 5 + 2))
            Sub_Data_Array[:,0:im_size] = ImData
            Sub_Data_Array2[:,0:im_size] = ImData_2

            for row, plate in enumerate(label):
                # padding plate
                plate[1] += left; plate[2] += top;
                plate[3] += left; plate[4] += top;

                # adding plate to the data vector
                Sub_Data_Array[row, im_size : im_size + 5] = plate
                Sub_Data_Array2[row, im_size: im_size + 5] = plate
                #Sub_Data_Array[im_size + 5*row+1:im_size + 5*row + 5] = plate


            ## checking if everything is correct
            #plate_tst = cv2.rectangle(ImData.reshape(100,200), (int(plate[0]), int(plate[1])), (int(plate[2]), int(plate[3])),
            #                         (200, 0, 0), 0)
            #plt.imshow(plate_tst)

            Sub_Data_Array[:,-2] = desired_x
            Sub_Data_Array[:,-1] = len(label)
            Sub_Data_Array2[:, -2] = desired_x
            Sub_Data_Array2[:, -1] = len(label)
            Sub_Data_Array = np.vstack([Sub_Data_Array, Sub_Data_Array2])

            # preparing external images
            exdata = np.zeros((len(label), im_size + 5 + 2))
            for no, lebimg in enumerate(eximgs):
                exdata[:im_size] = lebimg.reshape(1,-1)
                exdata[im_size:-2] = exlabel[no]
                Sub_Data_Array = np.vstack([Sub_Data_Array, exdata])
            #
            # The structure of sub_data_array is: Im_Matrix + all plates + number of plates
            #
            #########################################################################
            ###     Reading and writing data
            ###========================================================================

            if not os.path.isfile('R_CNN_inp_tar_data_c.h5'):
                 h5file = open_file("R_CNN_inp_tar_data_c.h5", mode="w", title="labeled_data")
                 gplate = h5file.create_group(h5file.root, "plate_data")
                 h5file.create_array(gplate, 'datray', Sub_Data_Array, "gplate")
                 h5file.close()
                 print('database is created by adding image "{}"'.format(file_name))
            # # if data file exits reading the content of the pytable data file
            else:
                h5file = open_file("R_CNN_inp_tar_data_c.h5","a")
                # programming for the general case
                # The step that I should do:
                Temp_Data_file = np.vstack([h5file.root.plate_data.datray[:], Sub_Data_Array])
                h5file.close()
                os.remove("R_CNN_inp_tar_data_c.h5")
                h5file = open_file("R_CNN_inp_tar_data_c.h5", mode="w", title="labeled_data")
                gplate = h5file.create_group(h5file.root, "plate_data")
                h5file.create_array(gplate, 'datray', Temp_Data_file, "gplate")
                h5file.close()
                print('image "{}" is successfully added to the database'.format(file_name))
                print('Data Base size is; ', Temp_Data_file.shape[0])
                del Sub_Data_Array, Temp_Data_file

        except: continue

print('Data is generated')