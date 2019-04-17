#
# the code is for extracting features
#
import numpy as np
import tensorflow as tf
import pandas as pd
from keras import applications
import argparse
import csv
import os
from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from sklearn import (decomposition, manifold, pipeline)
from keras.models import load_model



# ap = argparse.ArgumentParser(prog='Feature extractor')
# ap.add_argument('source', default='', help='Path to the source metadata file')
# ap.add_argument('img_path', default='', help='Path to the video files/(frame) file')
# ap.add_argument(
#     'model',
#     default='ResNet50',
#     nargs="?",
#     # type=named_model,
#     help='Name of the pre-trained model to use')
#
# /home/ali/CLionProjects/p_01/model_file
#
# arg = ap.parse_args()
#
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
class feature_extraction_n:
    def __init__(self, arg):
        # putting fix params here!
        # self.source_dir = os.path.dirname(arg.source)
        self.model_name = arg
        self.model = self.named_model()
    #
    def named_model(self):
        #
        if self.model_name == 'Xception':
            return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')
        if self.model_name == 'VGG16':
            return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
        if self.model_name == 'VGG19':
            return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
        if self.model_name == 'InceptionV3':
            return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        if self.model_name == 'MobileNet':
            # mmodel = load_model('./mobilenet_1_0_224_tf_no_top.h5')
            # return mmodel
            return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')
        return applications.resnet50.ResNet50(weights='imagenet', include_top=True, pooling='avg')
    #
    def get_feature(self, fimg):
        #
        # if os.path.isfile(fimg):
        #     print('is file: {}'.format(fimg))
        try:
                # load image setting the image size to 224 x 224
                # img = image.load_img(fimg, target_size=(224, 224))
                # in case I can use numpy to resize frames to the desired version
                img = np.resize(fimg, (224,224,3))
                # convert image to numpy array
                x = image.img_to_array(img)
                # the image is now in an array of shape (3, 224, 224)
                # but we need to expand it to (1, 3, 224, 224) as Keras is expecting a list of images
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                # extract the features
                features = self.model.predict(x)[0]
                # convert from Numpy to a list of values
                features_arr = np.char.mod('%f', features)
                return features_arr
        # TODO: extending for the batch version
        except Exception as ex:
                # skip all exceptions for now
                print(ex)
                pass
        # else:
        #     print('this frame is not a valid file')
        #     return None

class visualization:
    def __init__(self, name):
        self.name = name

    def named_model(self):
        if self.name == 'TSNE':
            # visualization with default parameters
            return manifold.TSNE(angle=0.5, early_exaggeration=12.0, init='random', learning_rate=200.0,
                                 method='barnes_hut', metric='euclidean', min_grad_norm=1e-07,
                                 n_components=2, n_iter=1000, n_iter_without_progress=300,
                                 perplexity=30.0, random_state=0, verbose=0)
        if self.name == 'PCA-TSNE':
            tsne = manifold.TSNE(
                random_state=0, perplexity=50, early_exaggeration=6.0)
            pca = decomposition.PCA(n_components=48)
            return pipeline.Pipeline([('reduce_dims', pca), ('tsne', tsne)])
        if self.name == 'PCA':
            return decomposition.PCA(n_components=48)
        raise ValueError('Unknown model')


# ------------------------------------------------------------------------
# def start():
#     try:
#         # read the source file
#         data = pd.read_csv(arg.source, sep='\t')
#         # extract features
#         features = map(get_feature, data.T.to_dict().values())
#         # remove empty entries
#         features = filter(None, features)
#
#         # write to a tab delimited file
#         source_filename = os.path.splitext(arg.source)[0].split(os.sep)[-1]
#         #
#         with open(os.path.join(source_dir, '{}_features.tsv'.format(source_filename)), 'w') as output:
#             w = csv.DictWriter(output, fieldnames=['id', 'features'], delimiter='\t', lineterminator='\n')
#             w.writeheader()
#             w.writerows(features)
#
#     except EnvironmentError as e:
#         print(e)


# if __name__ == '__main__':
#     start()





