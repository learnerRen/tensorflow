# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 22:37:58 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import tensorflow as tf
a1=tf.get_variable('a1',
                   shape=[1],
                   dtype=tf.float32,
                   initializer=tf.constant_initializer(1))
a2=tf.get_variable('a2',
                   shape=[1],
                   dtype=tf.float32,
                   initializer=tf.constant_initializer(2))
op=tf.add(a1,a2,name='add_op')
init=tf.global_variables_initializer()
saver=tf.train.Saver()
sess=tf.Session()
sess.run(init)
print(sess.run(op))
saver_path=saver.save(sess,"path/model.cpkt")
print("Model saved in file:", saver_path)
sess.close()