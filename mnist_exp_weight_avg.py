# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 22:40:02 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

input_node=784
layer1_node=500
output_node=10

def model(a0,w1,b1,w2,b2,exp_weight_avg=None):
    if exp_weight_avg==None:
        a1=tf.nn.relu(tf.matmul(a0,w1)+b1)
        return tf.matmul(a1,w2)+b2
    else:
        a1=tf.nn.relu(
                tf.matmul(a0,exp_weight_avg.average(w1))+exp_weight_avg.average(b1))
        '''
        exp_weight_avg.average we use this to average every value.
        when we define exp_weight_avg, 
        '''
        return tf.matmul(a1,exp_weight_avg.average(w2)+exp_weight_avg.average(b2))
    
def train(mnist,moving_average_decay=0.99,learning_rate=0.1,training_steps=1000,batch_size=128,learning_rate_decay=0.99,regularization_rate=0.01):
    x=tf.placeholder(tf.float32,[None,input_node],name='x-input')
    y_=tf.placeholder(tf.float32,[None,output_node],name='y-output')
    w1=tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev=0.1))
    b1=tf.Variable(tf.constant(0,dtype=tf.float32,shape=[layer1_node]))
    w2=tf.Variable(tf.truncated_normal([layer1_node,output_node],stddev=0.1))
    b2=tf.Variable(tf.constant(0,dtype=tf.float32,shape=[output_node]))
    global_step=tf.Variable(0,trainable=False)
    if moving_average_decay:
        exp_weight_avg=tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
        #print(tf.trainable_variables())
        variable_average_op=exp_weight_avg.apply(tf.trainable_variables())
        #variable_average_operation=exp_weight_avg.apply(tf.trainable_variables())
        #apply it on every trainable vraiabls. tf.trainable_variables() returns all trainable variables
        y=model(x,w1,b1,w2,b2,exp_weight_avg=exp_weight_avg)
    else:
        y=model(x,w1,b1,w2,b2,exp_weight_avg=None)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=y)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    regularizer=tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization=tf.add(regularizer(w1),regularizer(w2))
    loss=cross_entropy_mean+regularization
    learning_rate=tf.train.exponential_decay(learning_rate,
                                             global_step,
                                             mnist.train.images.shape[0]/batch_size,
                                             learning_rate_decay)
    '''
    we'll set decay_rate and trian_step
    the true learning_rate=learning_rate(orignal)*decay_rate^(global_step/decay_step)
    at first, global_step/decay_step (almost)=0  so learning_rate=orignal
    at final, global_step/decay_step (almost)=0  so learning_rate=learning_rate(orignal)*decay_rate
    namely, decay faster and faster
    '''
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    sess=tf.Session()
    init=tf.global_variables_initializer()
    sess.run(init)
    if moving_average_decay:
        train_op=tf.group([train_step,variable_average_op])
    else:
        train_op=train_step
    for i in range(training_steps):
        if i%1000==0:
            validate_accuracy=sess.run(accuracy,feed_dict={x:mnist.validation.images,y_:mnist.validation.labels})
            print("After {} traing step(s),validation accuracy on validation set:{}".format(i,validate_accuracy))
        xs,ys=mnist.train.next_batch(batch_size)
        sess.run(train_op,feed_dict={x:xs,y_:ys})
    
    test_accuracy=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
    print("Final accuracy on test set:{}".format(test_accuracy))
    sess.close()

mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

batch_size=100
learning_rate=0.8
learning_rate_decay=0.99
regularization_rate=0.0001
training_steps=10000
moving_average_decay=0.99

print("with exponential_average_weight")
train(mnist,
      moving_average_decay=moving_average_decay,
      learning_rate=learning_rate,
      training_steps=training_steps,
      learning_rate_decay=learning_rate_decay,
      regularization_rate=regularization_rate)
print("     ")

print("without exponential_average_weight")
train(mnist,
      moving_average_decay=0,
      learning_rate=learning_rate,
      training_steps=training_steps,
      learning_rate_decay=learning_rate_decay,
      regularization_rate=regularization_rate)
