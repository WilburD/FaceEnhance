# 本系统的深度学习神经网络模型
import tensorflow as tf
import numpy as np
import FaceInput

# 创建变量，存储在CPU中
def variable_on_cpu(name, shape, initializer, dtype):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

# 创建带方差的变量
def variable_with_stddev(name, shape, stddev, dtype):
    var = variable_on_cpu(name, 
        shape, 
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype),
        dtype)
    return var

# 卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 池化
def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
# 处理获取数据集
def inputs():
    images = FaceInput.get_trains()
    labels = FaceInput.get_labels()
    return images, labels

# 神经网络模型
def neural_networks_model(images):
    with tf.variable_scope('conv1') as scope:
        weights = variable_with_stddev('weights',
                                        shape = [5, 5, 3, 64],
                                        stddev = 5e-2,
                                        dtype = tf.float32)
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(images, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv1 = tf.nn.relu(a_conv, name=scope.name)
    h_pool1 = max_pool_2x2(z_conv1, 'pool1')

    with tf.variable_scope('conv2') as scope:
        weights = variable_with_stddev('weights',
                                        shape = [5, 5, 64, 128],
                                        stddev = 5e-2,
                                        dtype = tf.float32)
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool1, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv2 = tf.nn.relu(a_conv, name=scope.name)
    h_pool2 = max_pool_2x2(z_conv2, 'pool2')

    with tf.variable_scope('conv3') as scope:
        weights = variable_with_stddev('weights',
                                       shape=[5, 5, 128, 256],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool1, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv3 = tf.nn.relu(a_conv)
    h_pool3 = max_pool_2x2(z_conv3, 'pool3')

    with tf.variable_scope('conv4') as scope:
        weights = variable_with_stddev('weights',
                                       shape=[5, 5, 256, 512],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool1, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv4 = tf.nn.relu(a_conv)
    h_pool4 = max_pool_2x2(z_conv4, 'pool4')

    print(h_pool1.shape)
    print(h_pool2.shape)
    return None

X, Y = inputs()
X = tf.cast(X, tf.float32)
Y = tf.cast(Y, tf.float32)
neural_networks_model(X)
