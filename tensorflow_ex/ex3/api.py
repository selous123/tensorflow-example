#coding="utf-8"

#%%

import tensorflow as tf
import numpy as np

a = np.arange(0,10)
b = np.ones(shape=[10])

logits = tf.placeholder(dtype=tf.float32,shape=[10],name="logits")
labels = tf.placeholder(dtype=tf.float32,shape=[10],name="labels")


mul = tf.multiply(logits-0.5,labels-0.5)>0

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
mul_curr = sess.run(mul,feed_dict={
        logits:a,
        labels:b
        })


print mul_curr

#%%