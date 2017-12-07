import numpy as np
import tensorflow as tf
a = tf.random_uniform([5, 3], minval=0.0, maxval=10.0, dtype=tf.float32, name='a')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(a))

writer.close()