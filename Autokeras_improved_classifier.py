#
# the goal of this script is to improve the quality of autokeras trainer
#

import sys, os

from functools import reduce

import torch

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.preprocessor import OneHotEncoder, MultiTransformDataset
#


# # Autokeras model trainer
# model_trainer = ModelTrainer(model,
#                              loss_function=classification_loss,
#                              metric=Accuracy,
#                              train_data=train_data,
#                              test_data=test_data,
#                              verbose=True)
# 
# model_trainer.train_model(2, 1)
# model.eval()
# #




