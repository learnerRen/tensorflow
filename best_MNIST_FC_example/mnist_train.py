# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 00:43:54 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import os 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
batch_size=256
learning_rate_base=0.8
learning_rate_decay=0.99
regularization_rate=0.0001
epoch=3000
moving_average_decay=0.99
model_save_path="./path/model/"
model_name="model.ckpt"

def train(mnist):
    x=tf.placeholder(tf.float32,
                     [None,mnist_inference.INPUT_NODE],
                     name='x-input')
    y=tf.placeholder(tf.float32,
                     [None,mnist_inference.OUTPUT_NODE],
                     name='y-output')
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    y_hat=mnist_inference.inference(x,regularizer)
    global_step=tf.Variable(0,trainable=False)
    variable_average=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_average_op=variable_average.apply(tf.trainable_variables())
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y,1),logits=y_hat)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    learning_rate=tf.train.exponential_decay(
            learning_rate_base,
            global_step,
            mnist.train.images.shape[0]/batch_size,
            learning_rate_decay)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_op=tf.group(train_step,variable_average_op)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            xs,ys=mnist.train.next_batch(batch_size)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y:ys})
            if i%1000==0:
                print("After {} training step(s),loss is {}".format(i,loss_value))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
'''        
def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()        
'''
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
train(mnist)