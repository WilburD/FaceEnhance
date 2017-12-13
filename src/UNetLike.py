import tensorflow as tf

# 创建变量
def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, 
                            shape, 
                            initializer = initializer)
    return var

# 卷积操作函数
def conv2d(inputs, filter):
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    return tf.nn.conv2d(inputs, filter, strides, padding)

# 反卷积，上采样
def up_conv2d(inputs, filter, output_shape):
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    return tf.nn.conv2d_transpose(inputs, filter, output_shape, strides, padding)

# 池化，下采样操作
def max_pool_2x2(inputs):
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'SAME'
    return tf.nn.max_pool(inputs, ksize, strides, padding)

def neural_networks_model(images, batch_size, image_size):
    size1, size2, size3, size4 = image_size, int(image_size/2), int(image_size/4), int(image_size/8)

    # conv1
    with tf.variable_scope('conv1') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 3, 64],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        conv = conv2d(images, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv1 = tf.nn.relu(a_conv, name=scope.name)
    # pool1,下采样1
    with tf.variable_scope('pool1') as scope:
        h_pool = max_pool_2x2(z_conv1)
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 64, 128],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv2 = tf.nn.relu(a_conv, name=scope.name)
    # pool2，下采样2
    with tf.variable_scope('pool2') as scope:
        h_pool = max_pool_2x2(z_conv2)

    # conv3
    with tf.variable_scope('conv3') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 128, 256],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv3 = tf.nn.relu(a_conv, name=scope.name)
    # pool3，下采样3
    with tf.variable_scope('pool3') as scope:
        h_pool = max_pool_2x2(z_conv3)
    
    # conv4，下采样4
    with tf.variable_scope('conv4') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 256, 512],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv4 = tf.nn.relu(a_conv, name=scope.name)
    # pool4
    with tf.variable_scope('pool4') as scope:
        h_pool = max_pool_2x2(z_conv4)

    # conv5
    with tf.variable_scope('conv5') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [3, 3, 512, 1024],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        conv = conv2d(h_pool, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv5 = tf.nn.relu(a_conv, name=scope.name)

    # up_conv1，上采样1
    with tf.variable_scope('up_conv1') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 512, 1024],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        up_conv1 = up_conv2d(z_conv5, weights, [batch_size, size4, size4, 512])
    
    # up_conv2， 上采样2
    with tf.variable_scope('up_conv2') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 256, 512],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        inputs = tf.add(z_conv4, up_conv1)
        up_conv2 = up_conv2d(inputs, weights, [batch_size, size3, size3, 256])

    # up_conv3， 上采样3
    with tf.variable_scope('up_conv3') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 128, 256],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        inputs = tf.add(z_conv3, up_conv2)
        up_conv3 = up_conv2d(inputs, weights, [batch_size, size2, size2, 128])

    # up_conv4， 上采样4
    with tf.variable_scope('up_conv4') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 64, 128],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        inputs = tf.add(z_conv2, up_conv3)
        up_conv4 = up_conv2d(inputs, weights, [batch_size, size1, size1, 64])
    
    # conv6
    with tf.variable_scope('conv6') as scope:
        weights = variable_on_cpu('weights', 
                                shape = [5, 5, 64, 3],
                                initializer=tf.random_normal_initializer(mean=0, stddev=0.02))
        biases = variable_on_cpu('biases', [3], tf.constant_initializer(0.1))
        inputs = tf.add(z_conv1, up_conv4)
        conv = conv2d(inputs, weights)
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv6 = tf.nn.relu(a_conv, name=scope.name)

    output_images = tf.nn.sigmoid(z_conv6)
    return output_images
