# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 23:51:00 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""
import tensorflow as tf
INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def get_weight_variable(shape,regularizer):
    weights=tf.get_variable("weights",
                            shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",regularizer(weights))
        #add regularizer(weight) to the collection "lossed"(or the collection include "loss")
        #tf.get_collection("losses")
        #tf.add_n("losses") add all variables and return in the collection 
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope("layer1"):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable("biases",
                               [LAYER1_NODE],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    with tf.variable_scope("layer2"):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",
                               [OUTPUT_NODE],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0))
        layer2=tf.matmul(layer1,weights)+biases
    return layer2
