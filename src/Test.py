import tensorflow as tf

X = tf.random_normal(shape=[32, 256, 256, 3], mean=100, stddev=10)
Y = tf.random_normal(shape=[32, 256, 256, 3], mean=100, stddev=10)

def UNet(X):
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable(shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable(shape=[64], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding='SAME')
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv1 = tf.nn.relu(a_conv)

    with tf.variable_scope('pool1') as scope:
        h_pool = tf.nn.max_pool()