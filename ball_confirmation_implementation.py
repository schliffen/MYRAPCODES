#
# reading the frames and cropping the frames
#
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import time
#
# ----------------------------
#
frozen_graph_filename = '/home/ali/Rapsodo/ball_confirmation/variable_pb/output_graph.pb'
# frame address
frames = '/home/ali/CLionProjects/p_01/Vid_data/ba_20180423164701_its-repetition-fts_hc_sgsg/'
cropped_videos = '/home/ali/BASKETBALL_DATA/cropped_2/'

#
# def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Then, we import the graph_def into a new Graph and returns it
with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="prefix")
# return graph
# for op in graph.get_operations():
#     print(op.name)

x = graph.get_tensor_by_name('prefix/Placeholder:0')
y = graph.get_tensor_by_name('prefix/final_result:0')

# frlist = os.listdir(frames)
with open(frames + 'net_rect.txt', 'r') as f:
    coord = f.read().split(' ')



with tf.Session(graph=graph) as sess:
    # Note: we don't nee to initialize/restore anything
    # There is no Variables in this graph, only hardcoded constants

    for frm in sorted(os.listdir(frames)):
        if frm.split('.')[1] == 'txt': continue
        img = cv2.imread(frames + frm)
        # cropping small area around the hoop
        croped = img[int(np.round(float(coord[1])))-20:int(np.round(float(coord[1])))+ int(np.round(float(coord[3]))),
                int(np.round(float(coord[0]))):int(np.round(float(coord[0])))+int(np.round(float(coord[2])))]

        img = cv2.resize(croped, (128,128))
        start = time.time()
        y_out = sess.run(y, feed_dict={x: np.expand_dims(img, 0)})

        print('elapsed time:', time.time() - start, 'ball probability:', y_out)

        plt.imshow(img)
