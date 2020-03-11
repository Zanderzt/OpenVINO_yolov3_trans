#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : freeze_graph.py
#   Author      : YunYang1994
#   Created date: 2019-03-20 15:57:33
#   Description :
#
#================================================================


import tensorflow as tf
from core.yolov3 import YOLOV3

def concat_sml(detect_1,detect_2,detect_3):
    
    detect_1 = tf.identity(detect_1, name='detect_1')
    detect_2 = tf.identity(detect_2, name='detect_2')
    detect_3 = tf.identity(detect_3, name='detect_3')
    
    detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
    detections = tf.identity(detections, name='detections')
               
    return detections

def detections_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.
    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    center_x, center_y, width, height, attrs = tf.split(
        detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    y0 = center_y - h2
    x1 = center_x + w2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1, name="output_boxes")
    return detections


pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
output_node_names = ["inputs", "output_boxes"]

input_data = tf.placeholder(dtype=tf.float32, shape=[None, 416,416,3], name='inputs')

model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

detect = concat_sml(model.pred_sbbox, model.pred_mbbox, model.pred_lbbox)
boxse = detections_boxes(detect)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())




