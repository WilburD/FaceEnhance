# 本系统的深度学习神经网络模型
import tensorflow as tf
import numpy as np
import FaceInput
import matplotlib.pyplot as plt

# 创建变量，存储在CPU中
def variable_on_cpu(name, shape, initializer, dtype):
    # with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    # var = tf.Variable(tf.random_normal(shape=shape, mean=0, stddev=0.1))
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
def neural_networks_model(images, batch_size):
    # conv1
    with tf.variable_scope('conv1') as scope:
        weights = variable_with_stddev('weights',
                                        shape = [5, 5, 3, 64],
                                        stddev = 5e-2,
                                        dtype = tf.float32)
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(images, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        # z_conv = tf.nn.relu(a_conv, name=scope.name)
        z_conv = tf.nn.sigmoid(a_conv, name=scope.name)
    # pool1
    with tf.variable_scope('pool1') as scope:
        h_pool = max_pool_2x2(z_conv, 'pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = variable_with_stddev('weights',
                                        shape = [5, 5, 64, 128],
                                        stddev = 5e-2,
                                        dtype = tf.float32)
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        # z_conv = tf.nn.relu(a_conv, name=scope.name)
        z_conv = tf.nn.sigmoid(a_conv, name=scope.name)
    # pool2
    with tf.variable_scope('pool2') as scope:
        h_pool = max_pool_2x2(z_conv, 'pool2')
    
    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = variable_with_stddev('weights',
                                       shape=[5, 5, 128, 256],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        # z_conv = tf.nn.relu(a_conv, name=scope.name)
        z_conv = tf.nn.sigmoid(a_conv, name=scope.name)
    # pool3
    with tf.variable_scope('pool3') as scope:
        h_pool = max_pool_2x2(z_conv, 'pool3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        weights = variable_with_stddev('weights',
                                       shape=[5, 5, 256, 512],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        # z_conv = tf.nn.relu(a_conv, name=scope.name)
        z_conv = tf.nn.sigmoid(a_conv, name=scope.name)
    # pool4
    with tf.variable_scope('pool4') as scope:
        h_pool = max_pool_2x2(z_conv, 'pool4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        weights = variable_with_stddev('weights',
                                       shape=[5, 5, 512, 1024],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        biases = variable_on_cpu('biases', [1024], tf.constant_initializer(0.1), tf.float32)
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        # z_conv = tf.nn.relu(a_conv, name=scope.name)
        z_conv = tf.nn.sigmoid(a_conv, name=scope.name)
    # pool5
    with tf.variable_scope('pool5') as scope:
        h_pool = max_pool_2x2(z_conv, 'pool5')


    # deconv1 
    with tf.variable_scope('deconv1') as scope:
        kfilter = variable_with_stddev('filters',
                                       shape=[5, 5, 512, 1024],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        de_conv = tf.nn.conv2d_transpose(h_pool, 
                                        kfilter, 
                                        output_shape=[batch_size, 16, 16, 512],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')
    
    # deconv2
    with tf.variable_scope('deconv2') as scope:
        kfilter = variable_with_stddev('filters',
                                       shape=[5, 5, 256, 512],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        de_conv = tf.nn.conv2d_transpose(de_conv, 
                                        kfilter, 
                                        output_shape=[batch_size, 32, 32, 256],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')

    # deconv3
    with tf.variable_scope('deconv3') as scope:
        kfilter = variable_with_stddev('filters',
                                       shape=[5, 5, 128, 256],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        de_conv = tf.nn.conv2d_transpose(de_conv, 
                                        kfilter, 
                                        output_shape=[batch_size, 64, 64, 128],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')

    # deconv4
    with tf.variable_scope('deconv4') as scope:
        kfilter = variable_with_stddev('filters',
                                       shape=[5, 5, 64, 128],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        de_conv = tf.nn.conv2d_transpose(de_conv, 
                                        kfilter, 
                                        output_shape=[batch_size, 128, 128, 64],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')

    # deconv5
    with tf.variable_scope('deconv5') as scope:
        kfilter = variable_with_stddev('filters',
                                       shape=[5, 5, 3, 64],
                                       stddev=5e-2,
                                       dtype=tf.float32)
        de_conv = tf.nn.conv2d_transpose(de_conv, 
                                        kfilter, 
                                        output_shape=[batch_size, 256, 256, 3],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')
    output_images = de_conv
    # print(output_images)
    return output_images

def compute_loss(output_images, label_images):
    loss = tf.reduce_mean(tf.square(label_images - output_images))
    return loss

def train(input_images, label_images, image_sizes, iters, save_path):
    xs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    ys = tf.placeholder(tf.float32, [None, 256, 256, 3])
    
    output_images = neural_networks_model(xs, 16)
    cost_function = compute_loss(output_images, ys)
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cost_function)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iters):
            for t in range(0, image_sizes-16, 16):
                xs_batch, ys_batch = sess.run([input_images[t:t+16], label_images[t:t+16]])
                sess.run(train_step, feed_dict={xs: xs_batch, ys: ys_batch})
                if t % 1 == 0:
                    cost = sess.run(cost_function, feed_dict={xs: xs_batch, ys: ys_batch})
                    print('iters:%s, loss:%s' % (t, cost))
    saver.save(sess, save_path)
    sess.close()

X, Y = inputs()
X = tf.cast(X, tf.float32)
Y = tf.cast(Y, tf.float32)
print(X.shape)
# neural_networks_model(X[:32], 32)
save_path = '/home/wanglei/wl/face-enhance/model/model1.ckpt'
train(X, Y, 500, 10, save_path)