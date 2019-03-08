#
# Using autokeras model trainer
#
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import load_model
import pytest
import argparse
#
from autokeras.image.gan import Generator, Discriminator
from autokeras.nn.generator import CnnGenerator
from autokeras.nn.loss_function import classification_loss, regression_loss, binary_classification_loss
from autokeras.nn.metric import Accuracy, MSE
from autokeras.nn.model_trainer import ModelTrainer, GANModelTrainer


from keras.preprocessing.image import  ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=.1,
                                   height_shift_range=.1, shear_range=.2, zoom_range=.2, horizontal_flip=True,
                                   fill_mode='nearest',
                                   rescale=1./255,
                                   zca_whitening=False)
validation_datagen = ImageDataGenerator(
                                            rescale=1./255
                                         )
# -------------
# args = argparse.ArgumentParser()
# args.add_argument('-dt', '--trn_data',  default='/home/rapsodo/BASKETBALL/00_DATASETS/BG_subtraction_Data/train/', help='path to train dataset')
# args.add_argument('-dv', '--val_data',  default='/home/rapsodo/BASKETBALL/00_DATASETS/BG_subtraction_Data/valid/', help='path to val dataset')
# args.add_argument('-m', '--mdl',        default='in_out_model_2_1.h5', help='path to val dataset')
# args.add_argument('-d0', '--ddir',      default='./models/', help='path to val dataset')
# ap.add_argument('-bd', '--bdr', default='/home/ali/BASKETBALL_DATA/ball_noball/', help='ball no ball data')
# -------------
# ap = args.parse_args()





def test_model_trainer_classification(train_data, test_data, model):

    ModelTrainer(model,
                 train_data=train_data,
                 test_data=test_data,
                 metric=Accuracy,
                 loss_function=classification_loss,
                 verbose=True,
                 path='./test_keras_trainer/').train_model(max_iter_num=3)



def test_model_trainer_regression(train_data, test_data):
    model = CnnGenerator(1, (28, 28, 3)).generate().produce_model()
    ModelTrainer(model,
                 train_data=train_data,
                 test_data=test_data,
                 metric=MSE,
                 loss_function=regression_loss,
                 verbose=False,
                 path='./test_keras_trainer/').train_model(max_iter_num=3)



def test_gan_model_trainer(train_data):
    g_model = Generator(3, 100, 64)
    d_model = Discriminator(3, 64)

    GANModelTrainer(g_model, d_model, train_data, binary_classification_loss, True).train_model(max_iter_num=3)


def test_model_trainer_timout(train_data, test_data):
    model = CnnGenerator(3, (28, 28, 3)).generate().produce_model()
    timeout = 1
    with pytest.raises(TimeoutError):
        ModelTrainer(model,
                     train_data=train_data,
                     test_data=test_data,
                     metric=Accuracy,
                     loss_function=classification_loss,
                     verbose=True,
                     path='./test_keras_trainer/').train_model(max_iter_num=300, timeout=timeout)


def get_generated_data():
    train_generator = train_datagen.flow_from_directory(
        args.trn_data,
        target_size=(128, 128),
        batch_size=32,
        color_mode='rgb',
        shuffle=True,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        args.val_data,
        target_size=(128, 128),
        batch_size=16,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)


    trdata_X = []
    trdata_Y = []
    cntr = 0
    for image in train_generator:
        plt.imshow(image[0][0])
        trdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            trdata_Y.append(1)
        else:
            trdata_Y.append(0)
        cntr += 1
        if cntr == 2000:
            break

    tsdata_X = []
    tsdata_Y = []
    cntr = 0
    for image in validation_generator:
        # plt.imshow(image[0][0])
        tsdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            tsdata_Y.append(1)
        else:
            tsdata_Y.append(0)
        cntr += 1
        if cntr == 500:
            break

    trdata_X = np.array(trdata_X)
    trdata_Y = np.array(trdata_Y)
    tsdata_X = np.array(tsdata_X)
    tsdata_Y = np.array(tsdata_Y)

    return trdata_X, trdata_Y, tsdata_X, tsdata_Y




if __name__ == '__main__':

    # loading an example model here!

    tr_x, tr_y, ts_x, ts_y = get_generated_data()


    # loading model here

    model = load_model('in_out_model_2_1.h5')

    test_model_trainer_classification((tr_x, tr_y), (ts_x, ts_y), model)





