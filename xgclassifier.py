#
# classifier design
#
import numpy as np
from numpy import loadtxt
import xgboost as xg
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import sklearn as sk
#import keras as kr

# creating the model



# getting the data

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

#
X = dataset[:,:8]
Y = dataset[:,8]

seed=1
test_size = .33

x_train, y_train, x_test, y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)

# model

model = XGBClassifier()
model.fit(x_train, y_train)




# cross validation for detecting optimal parameters



