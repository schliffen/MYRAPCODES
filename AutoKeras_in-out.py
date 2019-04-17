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
from keras import applications
from keras.optimizers import SGD
import os.path as osp
from keras.models import load_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint, TensorBoard
import datetime





args = argparse.ArgumentParser()
args.add_argument("-d0", "--trn_data", default='/home/ali/BASKETBALL_DATA/in-out/tyfun/Train/',
                help="path to train dataset")
args.add_argument("-d1", "--val_data", default='/home/ali/BASKETBALL_DATA/in-out/tyfun/Test/',
                help="path to val dataset")
args.add_argument("-d2", "--root",   default = '/home/ali/CLionProjects/in_out_detection/NewApproach/', help='root path to the model')
args.add_argument("-d3", "--mname", default = 'newmodel_in-out', help = 'name of the model to be used')
args.add_argument("-d4", "--retrain", default = False, help = 'whether to retrain a model')
ap = args.parse_args()

from autokeras  import ImageClassifier
from keras.preprocessing.image import ImageDataGenerator
#
# data generators
#
train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                   rotation_range=30, width_shift_range=.3,
                                   height_shift_range=.3, shear_range=.3, zoom_range=.2, horizontal_flip=True,
                                   zca_whitening=False, zca_epsilon=1e-3, fill_mode='nearest'
                                   )
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    ap.trn_data,
    target_size=(128, 128),
    batch_size=4,
    class_mode='categorical',
    shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
    ap.val_data,
    target_size=(128, 128),
    batch_size=4,
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
            tsdata_Y.append(np.array([1, 0]))
        else:
            tsdata_Y.append(np.array([0, 1]))
        cntr +=1
        if cntr == num_data:
            break
    #
    tsdata_X = np.array(tsdata_X)
    tsdata_Y = np.array(tsdata_Y)
    #
    return tsdata_X, tsdata_Y
# -----------------------------------------------

def train_save(trdata_X, trdata_Y, tsdata_X, tsdata_Y, unique_name):

    # automatic training facility
    #
    clf = ImageClassifier(verbose=True, augment=True)
    clf.fit(trdata_X, trdata_Y, time_limit=6 * 60)
    clf.final_fit(trdata_X, trdata_Y, tsdata_X, tsdata_Y, retrain = False)

    # savig trained model
    y = clf.evaluate(tsdata_X, tsdata_Y)
    # I cant read this type of saving
    #clf.export_autokeras_model(ap.root + 'models/' + ap.mname + unique_name + str(y*100) + '.h5') # CANT load this type of saving
    clf.export_keras_model(ap.root + 'models/' + ap.mname + unique_name + str(y*100) + '.h5')
    ## the following is for loading the model
    # from autokeras.utils import pickle_from_file
    # model = pickle_from_file(model_file_name)
    # results = model.evaluate(x_test, y_test)
    # print(results)

# adding more options
def model_train_save(f_model, unique_name):

    # saving the model
    model_json = f_model.to_json()
    with open(ap.root + ap.mname + unique_name + " .json", "w") as json_file:
        json_file.write(model_json)

    weigth_file=ap.root + "weights/" + unique_name + "-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(weigth_file, monitor='val_acc', verbose=1, period=50) # save_best_only=True, mode='max'
    tbCallBack = TensorBoard(log_dir=ap.root + 'Graph/' + ap.mname + unique_name, histogram_freq=0,   write_graph=True, write_images=True)
    callbacks_list = [checkpoint,tbCallBack]
    history = f_model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples/train_generator.batch_size ,
        epochs = 1000,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        callbacks=callbacks_list,
        verbose=1)
    # serialize weights to HDF5
    f_model.save_weights(ap.root + 'models/' + ap.mname +  unique_name + ".h5")
    print("Saved model to disk")

    # savig trained model
    with open(ap.root + 'models/' + ap.mname +  unique_name + '.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    return history

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


def test_model(model_name, tsdata_X, tsdata_Y):

    model = load_model(ap.root + './models/' + model_name)
    model.summary()
    #
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    #
    # Evaluating with keras itself
    y = model.evaluate(tsdata_X, tsdata_Y)
    print("Evaluation results with keras: ", y * 100)
    # evaluating the model with
    y_prediction = model.predict(tsdata_X, verbose=1)
    result_score = accuracy_score(y_pred=y_prediction, y_true=tsdata_Y)
    print('Evaluating results with sklearn: ', result_score)


def convert2mlmodel(template, model_name):

    if template == 'single':
        # converting to coreml
        from keras.applications.mobilenet import relu6
        from keras.utils.generic_utils import CustomObjectScope
        from keras.models import model_from_json

        with CustomObjectScope({'relu6': relu6}):
            all_model = load_model(model_name + '.h5')
            # loading weights to the new model

        #
        coreml_model = coremltools.converters.keras.convert(all_model,
                                                            input_names="image",
                                                            image_input_names="image"
                                                            )
        coreml_model.save('./models/' +  model_name + '.mlmodel')

        print('model is saved successfully!')

    if template == 'multiple':
        js_file = open('./models/' + model_name + '.json', 'r')
        loaded_json_model = js_file.read()
        js_file.close()
        # converting to coreml
        from keras.applications.mobilenet import relu6
        from keras.utils.generic_utils import CustomObjectScope
        from keras.models import model_from_json

        with CustomObjectScope({'relu6': relu6}):
            loaded_model = model_from_json(loaded_json_model)
            # loading weights to the new model
            loaded_model.load_weights('./models/' + model_name +  '.h5')
        #
        coreml_model = coremltools.converters.keras.convert(loaded_model,
                                                            input_names="image",
                                                            image_input_names="image"
                                                            )
        coreml_model.save('./models/' + model_name + '.mlmodel')

        print('model is saved successfully!')


def retrain_model(model_name):
    #
    model = load_model('./models/' + model_name)
    #
    model.summary()
    # designing an optrimizer
    epochs = 1000
    learning_rate = 0.00001
    decay_rate = learning_rate/ epochs
    momentum = .91
    sgd = SGD(lr = learning_rate, momentum = momentum, decay = decay_rate, nesterov  = True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # specifying model specific name
    unique_start_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    print(unique_start_time)
    # X_train, Y_train = generate_train_data(2000)
    return model_train_save(model, unique_start_time)




if __name__ == '__main__':
    # getting test data
    # model representation
    # X_test, Y_test = generate_test_data(500)
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
    # if there exists a model, we can retrain the mdoel
    # looking for a proper model    # specifying model specific name

    if ap.retrain:
        unique_start_time = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
        print(unique_start_time)
        model_name = 'newmodel_in-out20190204_17464195.0.h5'
        history = retrain_model(model_name)
        # model representation
        illustrate(history)


    if True:
        trdata_X, trdata_Y = generate_train_data(500)
        tsdata_X, tsdata_Y = generate_train_data(100)
        train_save(trdata_X, trdata_Y, tsdata_X, tsdata_Y, unique_start_time)

    test = False
    if test:
        x_test, y_test = generate_test_data(500)
        model_name = 'newmodel_in-out20190205_153755100.0.h5'
        test_model(model_name, x_test, y_test)






    #


    #
    # convert2mlmodel('multiple', unique_start_time)



    # testing model
    # test_model(X_test, Y_test)


