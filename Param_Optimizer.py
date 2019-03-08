#
# this file is for traning parameter optimization purposes
#
from __future__ import print_function
import numpy as np
import pickle
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    with open('./data/X_train.pickle', 'rb') as file:
        x_train = pickle.load(file).astype('float32')
    with open('./data/Y_train.pickle', 'rb') as file:
        y_train = pickle.load(file).astype('float32')
    with open('./data/X_test.pickle', 'rb') as file:
        x_test = pickle.load(file).astype('float32')
    with open('./data/Y_test.pickle', 'rb') as file:
        y_test = pickle.load(file).astype('float32')

    x_train /= 255
    y_train /= 255
    x_test /= 255
    y_test /= 255

    # nb_classes = 2
    # y_train = np_utils.to_categorical(y_train, nb_classes)
    # y_test = np_utils.to_categorical(y_test, nb_classes)

    return x_train, y_train, x_test, y_test



def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(512, input_shape=(128,128,3)))
    model.add(Flatten())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([50, 100, 200])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))

        # We can also choose between complete sets of layers

        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss={{choice(['categorical_crossentropy', 'binary_crossentropy', 'MSLE'])}}, metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    result = model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=2,
              verbose=2,
              validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    a,b,c,d = data()

    if True:
        best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    # model = Sequential()
    # model.add(Dense(512, input_shape=(128, 128, 3)))
    # model.add(Flatten())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(100))
    # model.add(Dropout(0.5))
    # model.add(Activation('relu'))
    # model.add(Dense(2))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
    #               optimizer='adam')
    #
    # model.summary()
    #
    # pre = model.predict(a[0])
    # print('row model prediction: ', pre)
    #
    # X_train, Y_train, X_test, Y_test = data()
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(X_test, Y_test))
    # print("Best performing model chosen hyper-parameters:")
    # print(best_run)