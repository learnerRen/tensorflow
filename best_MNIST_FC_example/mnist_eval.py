# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 23:23:49 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train
eval_interval_secs=10
def evaluate(mnist):
    with tf.Graph().as_default():
        x=tf.placeholder(tf.float32,
                         [None,mnist_inference.INPUT_NODE],name='x-input')
        y=tf.placeholder(tf.float32,[None,10],name='y-input')
        y_hat=mnist_inference.inference(x,None)
        correct_prediction=tf.equal(tf.argmax(y_hat,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(mnist_train.moving_average_decay)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)
        i=0
        while i<(mnist_train.epoch/1000+1):
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_train.model_save_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,
                                             feed_dict={x:mnist.validation.images,
                                                        y:mnist.validation.labels})
                    print("After {} training step(s), validation accuracy={}".format(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return 
            time.sleep(eval_interval_secs)

mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
evaluate(mnist)