#
# This script is for design deep learning for new in out
#
import os, sys, glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
#
import cv2
import argparse
import keras
import autokeras
import pickle
import coremltools
import numpy as np
import tensorflow as tf
import talos as ta
import hyperas as hp
from keras import applications
from keras.layers import Activation
from keras.optimizers import SGD
import os.path as osp
from keras.models import load_model, Model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint, TensorBoard
import datetime
from autokeras.utils import pickle_from_file, pickle_to_file
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform




args = argparse.ArgumentParser()
args.add_argument("-d0", "--trn_data", default='/home/rapsodo/CLionProjects/in_out_detection/tyfun/Train/',
                help="path to train dataset")
args.add_argument("-d1", "--val_data", default='/home/rapsodo/CLionProjects/in_out_detection/tyfun/Test/',
                help="path to val dataset")
args.add_argument("-d2", "--root",   default = '/home/rapsodo/CLionProjects/in_out_detection/NewApproach/', help='root path to the model')
args.add_argument("-d3", "--mdir",   default = 'models/', help='root path to the model')
args.add_argument("-d4", "--mname", default = 'newmodel_in-out', help = 'name of the model to be used')
args.add_argument("-d5", "--retrain", default = True, help = 'whether to retrain a model')
ap = args.parse_args()

from autokeras  import ImageClassifier
from keras.preprocessing.image import ImageDataGenerator
#
# data generators
#
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                   rotation_range=30, width_shift_range=.3, rescale=1./255,
                                   height_shift_range=.3, shear_range=.3, zoom_range=.2, horizontal_flip=True,
                                   zca_whitening=False, zca_epsilon=1e-3, fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    ap.trn_data,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
    ap.val_data,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# ---------------

# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


def generate_train_data(num_data):
    trdata_X = []
    trdata_Y = []
    cntr = 0
    for image in train_generator:
        plt.imshow(image[0][0])
        trdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            trdata_Y.append(np.array([1, 0]))
        else:
            trdata_Y.append(np.array([0, 1]))
        cntr +=1
        if cntr == num_data:
            break
    trdata_X = np.array(trdata_X)
    trdata_Y = np.array(trdata_Y)
    #
    return trdata_X, trdata_Y,

def generate_test_data(num_data):
    le = preprocessing.LabelEncoder()
    le.fit([0, 1])
    print(le.classes_)
    #le.inverse_transform([0, 1])
    #le.transform()

    tsdata_X = []
    tsdata_Y = []
    cntr = 0
    for image in train_generator:
        # plt.imshow(image[0][0])
        tsdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            tsdata_Y.append(1) # np.array([1, 0])
        else:
            tsdata_Y.append(0)  #np.array([0, 1])
        cntr +=1 
        if cntr == num_data:
            break
    #
    tsdata_X = np.array(tsdata_X)
    tsdata_Y = np.array(tsdata_Y)
    #
    return tsdata_X, tsdata_Y

def data():
    trdata_X = []
    trdata_Y = []
    tsdata_X = []
    tsdata_Y = []
    cntr = 0
    for image in train_generator:
        trdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            trdata_Y.append(1)
        else:
            trdata_Y.append(0)
        cntr += 1
        if cntr == 200:
            break

    cntr = 0
    for image in train_generator:
        # plt.imshow(image[0][0])
        tsdata_X.append(image[0][0])
        if image[1][0][0] == 1:
            tsdata_Y.append(1)
        else:
            tsdata_Y.append(0)
        cntr +=1
        if cntr == 100:
            break
    #

    trdata_X = np.array(trdata_X)
    trdata_Y = np.array(trdata_Y)
    tsdata_X = np.array(tsdata_X)
    tsdata_Y = np.array(tsdata_Y)
    #
    return trdata_X, trdata_Y, tsdata_X, tsdata_Y

def Autokeras_data_flow():
    from autokeras.image.image_supervised import load_image_dataset

    # create data list as csv


    x_train, y_train = load_image_dataset(csv_file_path="train/label.csv",
                                          images_path="train")
    print(x_train.shape)
    print(y_train.shape)

    x_test, y_test = load_image_dataset(csv_file_path="test/label.csv",
                                        images_path="test")
    print(x_test.shape)
    print(y_test.shape)
    return x_train, x_test, y_train, y_test

# -----------------------------------------------

def model_post_process(model):

    # removing batchnormalization layer
    # not working for coreml conversion
    
    # for n_l, layer in enumerate(model.layers):
    #     lstr = str(layer).split('.')
    #     for i in range(len(lstr)):
    #         if len(lstr[i]) >= 18:
    #             if lstr[i][0:18] == 'BatchNormalization':
    #                 model.layers.pop(n_l)

    # x = model.output
    # x = Activation('softmax', name='activation_add')(x)
    # p_model = Model(model.input, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.save(ap.root + 'models/' + ap.mname + "_final_011.h5")

    return model


def search_good_model(trdata_X, trdata_Y, tsdata_X, tsdata_Y, unique_name):

    # automatic training facility
    #
    clf = ImageClassifier(verbose=True, augment=True)
    clf.fit(trdata_X, trdata_Y, time_limit= 5 * 60)
    clf.final_fit(trdata_X, trdata_Y, tsdata_X, tsdata_Y, retrain = False)

    # savig trained model
    y = clf.evaluate(tsdata_X, tsdata_Y)
    # I cant read this type of saving
    clf.export_autokeras_model(ap.root + 'models/' + ap.mname + unique_name + str(y*100) + '_01_.h5') # CANT be converted to keras and coreml
    saving_name = ap.root + 'models/' + ap.mname + unique_name + str(y * 100) + '_02_.h5'
    clf.export_keras_model(saving_name)
    # clf.load_searcher().load_best_model().produce_keras_model().save( ap.root + 'models/' + ap.mname + unique_name + str(y*100) + '_03_.h5' )

    print('accuracy: ', y*100 )
    return clf, saving_name


def model_optimization():
    best_run, best_model = optim.minimize(model=prepare_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    return best_run, best_model

def custom_load_model(model_name, template):
    if template == 'single':
        # converting to coreml
        from keras.applications.mobilenet import relu6
        from keras.utils.generic_utils import CustomObjectScope
        from keras.models import model_from_json

        with CustomObjectScope({'relu6': relu6}):
            model = load_model(model_name)
            # loading weights to the new model

    if template == 'multiple':
        js_file = open(model_name + '.json', 'r')
        loaded_json_model = js_file.read()
        js_file.close()
        # converting to coreml
        from keras.applications.mobilenet import relu6
        from keras.utils.generic_utils import CustomObjectScope
        from keras.models import model_from_json

        with CustomObjectScope({'relu6': relu6}):
            loaded_model = model_from_json(loaded_json_model)
            # loading weights to the new model
            loaded_model.load_weights( model_name + '.h5')

    return loaded_model

def retrain_model(model_name, model_template, unique_name):

    # reading the model
    f_model = custom_load_model(model_name, model_template)

    # adding softmax to the last layer
    x = f_model.output
    x = Activation('softmax', name='activation_add')(x)
    model = Model(f_model.input, x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    # saving the model itself
    model_json = model.to_json()
    with open(ap.root + 'models/' + ap.mname + unique_name + "retrained.json", "w") as json_file:
        json_file.write(model_json)

    weigth_file=ap.root + "weights/" + unique_name + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, period=20) # save_best_only=True, mode='max'
    tbCallBack = TensorBoard(log_dir=ap.root + 'Graph/' + ap.mname + unique_name, histogram_freq=0,   write_graph=True, write_images=True)
    # early_stopper(params['epochs'], mode=[1, 1])
    callbacks_list = [checkpoint, tbCallBack ]

    # optimization

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs = 100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        callbacks=callbacks_list,
        verbose=1)
    # # serialize weights to HDF5
    model.save_weights(ap.root + 'models/' + ap.mname +  unique_name + "retrained.h5")
    print("Saved model to disk")
    #
    # # savig trained model
    with open(ap.root + 'models/' + ap.mname +  unique_name + 'retrained.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    return model, history

def illustrate(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def test_model(model, tsdata_X, tsdata_Y):

    # Evaluating with keras itself
    y = model.evaluate(tsdata_X, tsdata_Y)
    print("Evaluation results with keras: ", y * 100)
    # evaluating the model with
    y_prediction = model.predict(tsdata_X)
    result_score = accuracy_score(y_pred=y_prediction, y_true=tsdata_Y)
    print('Evaluating results with sklearn: ', result_score)
    report = classification_report(y_true=tsdata_Y, y_pred=y_prediction)

    return report



def convert2mlmodel(model_name, template):
        #
        model = custom_load_model(model_name, template)
        pp_model = model_post_process(model)

        #
        coreml_model = coremltools.converters.keras.convert(pp_model,
                                                            input_names="image",
                                                            image_input_names="image",
                                                            image_scale=1./255
                                                            )
        coreml_model.save(model_name + '.mlmodel')




if __name__ == '__main__':

    # model name registration

    model_name = 'newmodel_in-out20190207_083415retrained'
    full_model_name = ap.root + ap.mdir + model_name
    model_template = 'multiple'

    unique_start_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    print(unique_start_time)

    if False:
        good_model = search_good_model(unique_start_time)


    if False:
        retr_model, history = retrain_model(full_model_name, model_template, unique_start_time)
        illustrate(history)

        #
        tsdata_X, tsdata_Y = generate_test_data(num_data)
        test_model(retr_model, tsdata_X, tsdata_Y)




    if False:
        c, d = generate_test_data(200)
        # auto designed
        model = pickle_from_file(full_model_name)
        report = test_model(model, c, d)
        with open('classification_report_' + model_name + '.pickle', 'wb') as f:
            pickle.dump(report, f)






    # POSTPROCESSING THE MODEL AND SAVE FOR FINAL USE
    # it returns post processed model
    if True:
        #
        # model = load_model(full_model_name)


        # SVG(model_to_dot(model).create(prog='dot', format='svg'))
        # model representation

        convert2mlmodel(full_model_name, 'multiple')


    #



    #
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, Y_test))
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)
    #
    # with open('best_model_01.pickle', 'wb') as f:
    #     pickle.dump(best_model, f)


    # trdata_X, trdata_Y = generate_train_data(500)
    # tsdata_X, tsdata_Y = generate_train_data(100)
    # train_save(trdata_X, trdata_Y, tsdata_X, tsdata_Y, unique_start_time)

    # test = True
    # if test:
    #     model_name = 'newmodel_in-out20190204_17464195.0.h5'
    #     test_model(model_name, tsdata_X, tsdata_Y)

    #


    # testing model
    # test_model(X_test, Y_test)


    # getting test data
    # model representation
    # X_test, Y_test = generate_test_data(500)
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    # if there exists a model, we can retrain the mdoel