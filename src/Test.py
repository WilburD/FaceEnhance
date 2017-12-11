import tensorflow as tf


X = tf.constant(1.0, shape=[1, 256, 256, 3])
print(X)

def t(X):
    with tf.variable_scope('conv1') as scope:
        w = tf.get_variable('weight', [5, 5, 3, 32], tf.constant(0.1))
        b = tf.get_variable('bias', [32], tf.constant(0.1))
        c = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    # print(sess.run(X))
    for i in range(1000):
        t(X)
sess.close()