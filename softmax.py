# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 23:58:25 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True) 
print("train images shape:{}".format(mnist.train.images.shape))
print("train label shape:{}".format(mnist.train.labels.shape))
print("test images shape:{}".format(mnist.test.images.shape))
print("test label shape:{}".format(mnist.test.labels.shape))
print("validation images shape:{}".format(mnist.validation.images.shape))
print("validation label shape:{}".format(mnist.validation.labels.shape))
x=tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=-tf.reduce_sum(y_*tf.log(y))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
'''
accuracy=0.918
'''