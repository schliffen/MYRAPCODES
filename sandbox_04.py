#
#
#
# from xgboost import XGBClassifier
# import lightgbm as lgbm
# import coremltools
import cv2
import numpy as np
import scipy
from scipy.misc import imread
import pickle
import random
import os
import matplotlib.pyplot as plt
import argparse
args = argparse.ArgumentParser()
args.add_argument('-d0', '--data', default='/home/ali/CLionProjects/rapsodo_ball_classification/data/in_out/test/', help='data directory')

ap = args.parse_args()
# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc


def batch_extractor(images_path, pickled_db_path="features.pck"):
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    for f in files:
        print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        result[name] = extract_features(f)

    # saving all our feature vectors in pickled file
    with open(pickled_db_path, 'w') as fp:
        pickle.dump(result, fp)

class Matcher(object):

    def __init__(self, pickled_db_path="features.pck"):
        with open(pickled_db_path) as fp:
            self.data = pickle.load(fp)
        self.names = []
        self.matrix = []
        for k, v in self.data.iteritems():
            self.names.append(k)
            self.matrix.append(v)
        self.matrix = np.array(self.matrix)
        self.names = np.array(self.names)

    def cos_cdist(self, vector):
        # getting cosine distance between search image and images database
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(self.matrix, v, 'cosine').reshape(-1)

    def match(self, image_path, topn=5):
        features = extract_features(image_path)
        img_distances = self.cos_cdist(features)
        # getting top 5 records
        nearest_ids = np.argsort(img_distances)[:topn].tolist()
        nearest_img_paths = self.names[nearest_ids].tolist()

        return nearest_img_paths, img_distances[nearest_ids].tolist()

def show_img(path):
    img = imread(path, mode="RGB")
    plt.imshow(img)
    plt.show()
    return img

def run():
    files = [os.path.join(ap.data + 'in/' , p) for p in sorted(os.listdir(ap.data + 'in/' ))]
    # getting 3 random images
    sample = random.sample(files, 3)

    # batch_extractor(images_path)

    # ma = Matcher('features.pck')

    for s in sample:
        print('Query image ==========================================')
        img = show_img(s)
        eq = []
        channels = 3

        # p1 histogram equalization
        for i in range(channels):
            eq.append(cv2.equalizeHist(img[:,:,i]))
        img_eq = np.transpose(np.array(eq), (1,2,0))
        # eqc = np.hstack([eq[i] for i in range(channels)])
        # p2 img sharpenning
        laplacianBoostFactor = 1.2
        # kern = np.array([[0, -1, 0], [-1, 5*laplacianBoostFactor, -1], [0, -1, 0]])
        # img_flt = cv2.filter2D(img, img.shape[2], kernel=kern)
        # img_blr = cv2.GaussianBlur(img, (5, 5), 0.67, 0.67)
        # unsharpMask = img - img_blr
        # unsharpMask = cv2.threshold(unsharpMask, 20, 255, cv2.THRESH_TOZERO)
        # unsharpBoostFactor = 9
        # unsharpSharpening = img + (unsharpMask * unsharpBoostFactor)
        # cv2.imshow("Histogram Equalization", np.hstack([img, img_eq, unsharpMask, img_blr]))
        # cv2.waitKey(0)

        # testing canny edge
        canny = cv2.Canny(img_eq, 30, 150)




        #hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        #plt.figure()
        #plt.title("Grayscale Histogram")
        #plt.xlabel("Bins")
        #plt.ylabel("# of Pixels")
        #plt.plot(hist)
        #plt.xlim([0, 256])
        #plt.show()
        #cv2.waitKey(0)
        #
        # names, match = ma.match(s, topn=3)
        # print('Result images ========================================')
        # for i in range(3):
            # we got cosine distance, less cosine distance between vectors
            # more they similar, thus we subtruct it from 1 to get match value
            # print('Match %s' % (1-match[i]))
            # show_img(os.path.join(images_path, names[i]))

# if 'name' == '__main__':
run()