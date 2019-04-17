#
# The Goal is to improve training images augmentation and equialization of the classes
#
# Final Part of Data Creation - This is developed by ALI for RAPSODO
#
import numpy as np
import _pickle as pickle
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import  ImageDataGenerator

aug = ImageDataGenerator(rotation_range=30, width_shift_range=.1,
                         height_shift_range=.1, shear_range=.2, zoom_range=.2, horizontal_flip=True,
                         fill_mode='nearest', zca_whitening=True)
#
# Reading and splitting the data
save_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/'
data_dir = '/home/ali/BASKETBALL_DATA/Frames_sorted_by_shot/'
data = ['data_02/video_ba_03_data-02.pickle', 'data_03/video_ba_03_data-03.pickle']
label =  ['data_02/video_ba_03_label-02.pickle', 'data_03/video_ba_03_label-03.pickle']


gold_data = []
gold_label = []

for (ditem, litem) in zip(data, label):
    with open(data_dir + ditem, 'rb') as f:
        gold_data.append(pickle.load(f))
    with open(data_dir + litem, 'rb') as f:
        gold_label.append(pickle.load(f))

# choosing same number positive and negative
# 1. numbe rof outs

# trX, tsX, trY, tsY = train_test_split(gold_data, gold_label, train_size=1)

total_out = 0
total_in = 0
redundant = 0
for d in range(len(gold_label)):
    for i in range(gold_label[d].shape[0]):
        if gold_label[d][i] == 0:
            total_out += 1

        elif gold_label[d][i] == 1:
            total_in  += 1
        else:
            redundant += 1

print('total in shots: %2f%% total out shots: %2f%% total redundant data: %2f%%' % (total_in, total_out, redundant))
# 2. random selection of data
inp_data = []
inp_label = []
#
nin = 0
nout=0
# setting number of data to be created:
total_data = 1000

while (nout < total_out) or (nin < total_out):
    for d in range(len(gold_data)):
        rind = np.random.randint(0,gold_label[d].shape[0],1)
        if gold_label[d][rind] == 0 and (nout < total_out):
            nout +=1
            if len(gold_data[d][rind][0]) < 40:
                for j in range(40-len(gold_data[d][rind][0])):
                    gold_data[d][rind][0].insert(0, np.zeros(28*28*3))
            for j in range(40):
                inp_data.append(gold_data[d][rind][0][j])
            inp_label.append(gold_label[d][rind][0])

        if (gold_label[d][rind] == 1) and (nin < total_out):
            nin +=1
            if len(gold_data[d][rind][0]) < 40:
                for j in range(40-len(gold_data[d][rind][0])):
                    gold_data[d][rind][0].insert(0, np.zeros(28*28*3))
            for j in range(40):
                inp_data.append(gold_data[d][rind][0][j])
            inp_label.append(gold_label[d][rind][0])



del gold_label, gold_data
# saving this randomly equalized data
with open(save_dir + 'Data_equalized_input_181009_.pickle', 'wb') as f:
    pickle.dump(np.array(inp_data), f, protocol=4)
with open(save_dir + 'Data_equalized_label_181009_.pickle', 'wb') as f:
    pickle.dump(np.array(inp_label), f, protocol=4)

print('data %1f label %f' %(len(inp_data), len(inp_label)))
