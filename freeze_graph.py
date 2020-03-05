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
def detections_boxes(detections, detectionm, detectionl):
    
    detec = tf.concat([detections,detectionm,detectionl], axis = 1,name="output_boxes")
    return detec

pb_file = "./yolov3_coco.pb"
ckpt_file = "./checkpoint/yolov3_coco_demo.ckpt"
output_node_names = ["inputs", "output_boxes"]

# with tf.name_scope('input'):
#    input_data = tf.placeholder(dtype=tf.float32, name='input_data')
input_data = tf.placeholder(dtype=tf.float32, shape=[1, 416, 416, 3], name="inputs")

with tf.variable_scope('detector'):
    model = YOLOV3(input_data, trainable=False)
print(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

detection = detections_boxes(model.conv_sbbox, model.conv_mbbox, model.conv_lbbox)

sess  = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess, ckpt_file)

converted_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                            input_graph_def  = sess.graph.as_graph_def(),
                            output_node_names = output_node_names)

with tf.gfile.GFile(pb_file, "wb") as f:
    f.write(converted_graph_def.SerializeToString())



