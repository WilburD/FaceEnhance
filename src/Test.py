import tensorflow as tf

def the_variable(X):
    Weights = tf.Variable(tf.random_normal(shape=[5, 5, 3, 64], mean=0, stddev=0.01), name='Weights')
    biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
    conv = tf.nn.conv2d(X, Weights, strides=[1, 1, 1, 1], padding='SAME', name='conv')
    a_conv = tf.nn.bias_add(conv, biases, name='bias_add')
    z_conv = tf.nn.relu(a_conv, name='relu')
    return z_conv

X = tf.constant(1.0, shape=[10, 256, 256, 3], name='X')
with tf.Session() as sess:
    for i in range(10000):
        Z = the_variable(X)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
writer.close()