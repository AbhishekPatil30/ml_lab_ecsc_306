
# coding: utf-8

# In[6]:

import tensorflow as tf


with tf.name_scope('Input'):
    y=tf.constant([20.63,11.2,12.34,10.3,9.2,14.3])
    x=tf.constant([17.2,17.9,18.4,25.4,16.3,12.21])

with tf.name_scope('mean'):

    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)

with tf.name_scope('var_x'):
 
    x_vara = tf.subtract(x,x_mean)
    x_varb = tf.square(x_vara)
    x_variance = tf.reduce_sum(x_varb)

with tf.name_scope('var_y'):

    y_vara = tf.subtract(y,y_mean)
    y_varb = tf.square(y_vara)
    y_variance = tf.reduce_sum(y_varb)

with tf.name_scope('covariance'):

    covariance1 = tf.multiply(x_vara,y_vara)
    covariance2 = tf.reduce_sum(covariance1)
    covariance = tf.div(covariance2,5)

with tf.name_scope('slope'):

    m=tf.div(covariance,x_variance)

with tf.name_scope('intercept'):
    
    c1=tf.multiply(m,x_mean)
    c=tf.subtract(y_mean,c1)

with tf.Session() as sess1:
    writer = tf.summary.FileWriter("/tmp/tboard/parta", sess1.graph)
    print(sess1.run(m))
    print(sess1.run(c))
    writer.close()

