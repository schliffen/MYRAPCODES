#
#
#
import sys, os, glob
import tensorflow as tf
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

#


args = argparse.ArgumentParser()
args.add_argument('-d1', '--ddir', type=str, default='/home/ali/BASKETBALL_DATA/test/', help='get data for test')
args.add_argument('-d2', '--mdir', type=str, default="/home/ali/CLionProjects/rapsodo_ball_detection/FrozenGraph/lite_model/tflite_ball_1600.tflite", help='get data for test')
args.add_argument('-t', '--tnm',   type=str, default='net_rect.txt',help='the text file for hoop coordinates')

arg = args.parse_args()

# --------------------- interpretor results ---------------------------------------
# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path=arg.mdir)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test model on random input data.
input_shape = input_details[0]['shape']
#
# ---------------- loading images ------------------------


# ------------------------------------------------
# the process of detecting the ball
# ------------------------------------------------
# function for representation
def sub_vis_cv2_2(img, bbox,  scores, threshold, thickness):
    # color = colors[classes[i]]
    # Draw bounding box...
    shape = img.shape
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    # Draw text...
    if (scores > threshold):
        class_name = 'ball'
        color = [112,100,255]
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        display_str = ' {} : {} % '.format(str(class_name), int(100 * scores))
        cv2.putText(img, display_str, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 1, color, 3)
    return img

class decision_maker:
    def __init__(self, hoop_coord, shape):
        self.cover_threshold = .5
        self.mx_ball_area = 400
        self.mn_ball_area = 300
        self.hoop_coord = hoop_coord
        self.p1 = 0
        self.p2 = 0 #(int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        self.shape = shape
        self.dis_thresh = 10

    def iou(self, coord_1, coord_2):
        # computing iou between two rectangles
        x1_t, y1_t, x2_t, y2_t = coord_2
        x1_p, y1_p, x2_p, y2_p = self.p1[0], self.p1[1], self.p2[0], self.p2[1]
        if (x1_p > x2_p) or (y1_p > y2_p):
            raise AssertionError(
                "Prediction box is malformed? pred box: {}".format(coord_1))
        if (x1_t > x2_t) or (y1_t > y2_t):
            raise AssertionError(
                "Ground Truth box is malformed? true box: {}".format(coord_2))
        if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
            return 0.0
        far_x  = np.min([x2_t, x2_p])
        near_x = np.max([x1_t, x1_p])
        far_y  = np.min([y2_t, y2_p])
        near_y = np.max([y1_t, y1_p])
        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
        pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
        return iou

    def ball_computation(self):
        area = (self.p2[1]-self.p1[1])*(self.p2[0]-self.p1[0])
        center = [(self.p1[0] + self.p2[0])/2, (self.p1[1] + self.p2[1])/2]
        return area, center

    def distance_measure(self, center_1, center_2):
        return np.sqrt((center_1[1]-center_2[1])**2 + (center_1[0] - center_2[0])**2 )

    def ball_assessment(self):
        b_area = self.ball_computation()[0]
        #
        if b_area < self.mx_ball_area and b_area > self.mn_ball_area:
            return 1
        else:
            return 0
    # this function is to decide whether ball is in or out
    def inout(self, ball_coord):
        # do some basic computations here
        ball = False
        for i in range(3):
            self.p1 = (ball_coord[i][0] * self.shape[0], ball_coord[i][1] * self.shape[1])
            self.p2 = (ball_coord[i][2] * self.shape[0], ball_coord[i][3] * self.shape[1])
        #
            if self.ball_assessment():
                ball = True
                break
        if not ball:
            return 2
        #
        hoop_center = [(self.hoop_coord[0] + self.hoop_coord[2])/2, (self.hoop_coord[1] + self.hoop_coord[3])/2]
        ball_center = [(self.p1[0] + self.p2[0])/2, (self.p1[1] + self.p2[1])/2]
        distance = self.distance_measure(hoop_center, ball_center)


        if distance < self.dis_thresh:
            # if distance < self.dis_thresh:
            coverance = self.iou(ball_coord, self.hoop_coord)
            #
            if coverance > self.cover_threshold:
                return 1
            # later I will can another decision maker here
            # dicision maker
        elif ball_center[1] > hoop_center[1]:
            return 2
        else:
            return 0


# we already have the coordinates
# with open(arg.ddir + arg.tnm, 'r') as f:
#     coord = f.read().split(' ')
# img_ratio = [130/180, 130/213]
# hoop_coord = [int(np.round(float(coord[0]))), int(np.round(float(coord[1])))* img_ratio[1], int(np.round(float(coord[2]))),
#               int(np.round(float(coord[3])))]

hoop_coord = [61, 51, 89, 81]
img_list = sorted(glob.glob(arg.ddir + '*.png'))
for item in img_list:
    # start to get results
    img = cv2.imread(item)
    img = cv2.resize(img, (130,130))
    input_img = np.reshape(img, input_shape).astype(np.float32)[0]
    #
    input_img = (input_img - 128)/128
    # doing the process
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(input_img,0))
    interpreter.invoke()
    # getting the result
    output_coord   = interpreter.get_tensor(output_details[0]['index'])
    output_classes = interpreter.get_tensor(output_details[1]['index'])
    output_scores  = interpreter.get_tensor(output_details[2]['index'])
    #
    sub_vis_cv2_2(img, output_coord[0][0], output_scores[0][0], .1, 1)
    plt.imshow(img)
    #
    make_decision = decision_maker(hoop_coord, input_shape[1:3])
    result = make_decision.inout(output_coord[0])

    if result==1:
        print('IN')
    elif result ==2:
        continue
    else: print('OUT!')

    # see the true results



# change the following line to feed into your own data.





# print(output_scores)