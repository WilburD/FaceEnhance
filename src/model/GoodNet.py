import tensorflow as tf
from model import UNetLike

# fully convolution networks
class GoodNet:
    # 初始化数据集
    def __init__(self, images, labels):
        """Initial dataset

        Args:
            images: Tesor, some images to train,         shape = [batch_size, width, height, RGB_channels]
            labels: Tesor, some labels to compute loss,  shape = [batch_size, width, height, RGB_channels]
        Return:
            nothing
        """

        self.images = images
        self.labels = labels

    # 创建变量
    def variable_on_cpu(self, name, shape, initializer):
        """Helper to create variable

        Args:
            name: value is the name, for instance, name = 'weight'
            shape: variable's shape, for instance, shape = [5, 5, 32, 64]
            initializer: the initializer to get value
        Return:
            variable: Tensor, variable

        """
        var = tf.get_variable(name = name,
                            shape = shape,
                            initializer = initializer, 
                            dtype = tf.float32)
        return var
    

    # 创建卷积层
    def conv_layer(self, inputs, window_size, input_channals, output_channals):
        """Helper to add layer to model

        Args:
            inputs: layer's inputs
            window_size: value is filter's window size
            input_channals: 
            output_channals:
        Return:
            z_conv: Tensor, conv layer outputs
            weights:
            biases:
        """
        weights = self.variable_on_cpu('wights',
                                shape = [window_size, window_size, input_channals, output_channals],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02))
        biases = self.variable_on_cpu('biases', [output_channals], tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv = tf.nn.relu(a_conv)
        return z_conv, weights, biases

    # 创建池化层
    def pool_layer(self, inputs):
        """Hepler to pool 2x2
        
        Args:
            x: Tensor, from the conv layer
        Return:
            pool: Tensor, max_pool

        """
        return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 创建去卷积层
    def deconv_layer(self, inputs, window_size, batch_size, width, height, input_channals, output_channals):
        """Helper to create deconv layer

        deconv lyaer outputs' shape = [batch_sizem width, height, output_channals]

        Args:
            inputs: Tensor, inputs
            window_size: value is filter's window size
            batch_size: value is trian batch's size
            width: outputs' width
            height: outputs' height
            input_channals: 
            output_channals
        Return:
            deconv: Tensor, deconv layer outputs
            kfilters:

        """

        kfiters = self.variable_on_cpu('filters',
                                shape = [window_size, window_size, output_channals, input_channals],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02))
        deconv = tf.nn.conv2d_transpose(inputs,
                                        kfiters,
                                        output_shape = [batch_size, width, height, output_channals],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')
        return deconv, kfiters

    # Coarse网络结构
    def coarse_net_model(self, batch_size, width, height):
        """Build the GoodNet model

        Args:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels

        """
        predict_images, coarse_parms = self.u_net_2_model(batch_size, width, height)

        return predict_images, coarse_parms

    # Coarse loss function
    def coarse_loss(self, coarse_images):
        """compute the Coarse Stage loss

        Args:
            coarse_images: Tensor, the coarse stage predict_images
        Returns:
            coares_loss: value is coares stage loss
            
        """

        coares_loss = tf.reduce_mean(tf.square(self.labels - coarse_images))
        return coares_loss

    # Fine网络结构
    def fine_net_model(self, coarse_images, batch_size, width, height):
        """Build th U-Net model

        Args:
            coarse_images:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels

        """

        # predict_images, fine_parms = UNetLike.neural_networks_model(coarse_images, batch_size, width, height)
        predict_images, fine_parms = self.sr_net_model()
        return predict_images, fine_parms

    # Fine loss function
    def fine_loss(self, fine_images):
        """compute the Fine Stage loss

        Args:
            fine_images: Tensor, the coarse stage predict_images
        Returns:
            fine_loss: value is coares stage loss
            
        """

        # fine_loss1 = tf.reduce_mean(tf.square(self.labels[:, 100:200, 120:200] - fine_images[:, 100:200, 120:200]))
        fine_loss2 = tf.reduce_mean(tf.square(self.labels[:, 0:244, 0:244] - fine_images))
        return fine_loss2

    # 子网络结构: 2次下采样U-Net
    def u_net_2_model(self, batch_size, width, height):
        """Build the GoodNet model

        Args:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels
            u_net_2_parms

        """

        width1, width2 = width, int(width/2)
        height1, height2 = height, int(height/2)

        u_net_2_parms = []

        # conv11
        with tf.variable_scope('u_net2_conv11') as scope:
            z_conv11, weights, biases = self.conv_layer(self.images, 3, 3, 64)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv12
        with tf.variable_scope('u_net2_conv12') as scope:
            z_conv12, weights, biases = self.conv_layer(z_conv11, 3, 64, 64)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)

        # pool1 下采样1
        with tf.variable_scope('u_net2_pool1') as scope:
            h_pool1 = self.pool_layer(z_conv12)
        
        # conv21
        with tf.variable_scope('u_net2_conv21') as scope:
            z_conv21, weights, biases = self.conv_layer(h_pool1, 3, 64, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv22
        with tf.variable_scope('u_net2_conv22') as scope:
            z_conv22, weights, biases = self.conv_layer(z_conv21, 3, 128, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        
        # pool2 下采样2
        with tf.variable_scope('u_net2_pool2') as scope:
            h_pool2 = self.pool_layer(z_conv22)
        
        # conv31
        with tf.variable_scope('u_net2_conv31') as scope:
            z_conv31, weights, biases = self.conv_layer(h_pool2, 3, 128, 256)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv32
        with tf.variable_scope('u_net2_conv32') as scope:
            z_conv32, weights, biases = self.conv_layer(z_conv31, 3, 256, 256)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)

        # deconv1 上采样1
        with tf.variable_scope('u_net2_deconv1') as scope:
            deconv1, kfilters = self.deconv_layer(z_conv32, 3, batch_size, width2, height2, 256, 128)
            u_net_2_parms.append(kfilters)

        # conv41
        with tf.variable_scope('u_net2_conv41') as scope:
            inputs = tf.concat([z_conv22, deconv1], axis=3)
            z_conv41, weights, biases = self.conv_layer(inputs, 3, 256, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv42
        with tf.variable_scope('u_net2_conv42') as scope:
            z_conv42, weights, biases = self.conv_layer(z_conv41, 3, 128, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)

        # deconv2 上采样2
        with tf.variable_scope('u_net2_deconv2') as scope:
            deconv2, kfilters = self.deconv_layer(z_conv42, 3, batch_size, width1, height1, 128, 64)
            u_net_2_parms.append(kfilters)
            
        # conv51
        with tf.variable_scope('u_net2_conv51') as scope:
            inputs = tf.concat([z_conv12, deconv2], axis=3)
            z_conv51, weights, biases = self.conv_layer(inputs, 5, 128, 64)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv52
        with tf.variable_scope('u_net2_conv52') as scope:
            z_conv52, weights, biases = self.conv_layer(z_conv51, 3, 64, 64)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        # conv53
        with tf.variable_scope('u_net2_conv53') as scope:
            z_conv53, weights, biases = self.conv_layer(z_conv52, 3, 64, 3)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        
        predict_images = z_conv53
        return predict_images, u_net_2_parms

    # 子网络结构: SRCNN 超分辨率卷积网络
    # Dong C, Chen C L, He K, et al. Image Super-Resolution Using Deep Convolutional Networks[J].
    #  IEEE Transactions on Pattern Analysis & Machine Intelligence, 2016, 38(2):295.
    def sr_net_model(self):
        """Build the SRCNN model

        Args:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels
            sr_net_parms:
            
        """

        sr_net_parms = []

        # conv1, patch extraction and representation
        with tf.variable_scope('srcnn_conv1') as scope:
            weights = self.variable_on_cpu('wights',
                                shape = [9, 9, 3, 64],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.005))
            biases = self.variable_on_cpu('biases', [64], tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(self.images, weights, strides=[1, 1, 1, 1], padding='VALID')
            a_conv = tf.nn.bias_add(conv, biases)
            z_conv1 = tf.nn.relu(a_conv)
            sr_net_parms.append(weights)
            sr_net_parms.append(biases)

        # conv2, non-liner mapping
        with tf.variable_scope('srcnn_conv2') as scope:
            weights = self.variable_on_cpu('wights',
                                shape = [1, 1, 64, 32],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.005))
            biases = self.variable_on_cpu('biases', [32], tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(z_conv1, weights, strides=[1, 1, 1, 1], padding='VALID')
            a_conv = tf.nn.bias_add(conv, biases)
            z_conv2 = tf.nn.relu(a_conv)
            sr_net_parms.append(weights)
            sr_net_parms.append(biases)

        #conv3, reconstruction
        with tf.variable_scope('srcnn_conv3') as scope:
            weights = self.variable_on_cpu('wights',
                                shape = [5, 5, 32, 3],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.005))
            biases = self.variable_on_cpu('biases', [3], tf.constant_initializer(0.01))
            conv = tf.nn.conv2d(z_conv2, weights, strides=[1, 1, 1, 1], padding='VALID')
            a_conv = tf.nn.bias_add(conv, biases)
            z_conv3 = tf.nn.relu(a_conv)
            sr_net_parms.append(weights)
            sr_net_parms.append(biases)

        predict_images = z_conv3

        return predict_images, sr_net_parms
