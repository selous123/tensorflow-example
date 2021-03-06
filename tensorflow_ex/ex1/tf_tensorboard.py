#coding:utf-8


#######softmax

import tensorflow as tf
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data


######prepare data
mnist = input_data.read_data_sets('MNIST_data/',one_hot = True);


#######create the graph

x = tf.placeholder(tf.float32,shape = [None,784],name = 'x');
y_ = tf.placeholder(tf.float32,shape = [None,10],name = 'y_');

#initialize weights and bias;
#with tf.name_scope('input_weight'):
W = tf.Variable(tf.zeros([784,10]));

#with tf.name_scope('input_bias'):
b = tf.Variable(tf.zeros([10]),name = 'input_bias');

#Predict class and loss function
#with tf.name_scope('y'):
y = tf.nn.softmax(tf.matmul(x,W) + b)
tf.summary.histogram('y',y);
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_));
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = y_))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
tf.summary.scalar('loss_function', cross_entropy)

#train_step
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy);
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#####optimization:梯度下降
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



#Cannot evaluate tensor using `eval()`: No default session is registered. 
#Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`

#set sess as default
sess = tf.InteractiveSession();





########create session
#sess = tf.Session();
#init op 
init = tf.global_variables_initializer();
sess.run(init);
####TensorBoard
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/mnist_logs',sess.graph)


#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
    batch = mnist.train.next_batch(100);
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]});
    summary_str = sess.run(merged_summary_op,feed_dict={x: batch[0], y_: batch[1]});
    summary_writer.add_summary(summary_str, i);
    
    if i % 50 == 0:
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
        print "Setp: ", i, "Accuracy: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        



########evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels}))