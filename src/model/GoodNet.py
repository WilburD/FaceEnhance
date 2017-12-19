import tensorflow as tf
import UNetLike


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
                            initializer = initializer)
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

        """
        weights = self.variable_on_cpu('wights',
                                shape = [window_size, window_size, input_channals, output_channals],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02))
        biases = self.variable_on_cpu('biases', [output_channals], tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        a_conv = tf.nn.bias_add(conv, biases)
        z_conv = tf.nn.relu(a_conv)
        return z_conv

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

        """

        kfiters = self.variable_on_cpu('filters',
                                shape = [window_size, window_size, output_channals, input_channals],
                                initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02))
        deconv = tf.nn.conv2d_transpose(inputs,
                                        kfiters,
                                        output_shape = [batch_size, width, height, output_channals],
                                        strides = [1, 2, 2, 1],
                                        padding = 'SAME')
        return deconv

    # 网络1结构
    def good_net_model(self, batch_size, width, height):
        """Build the GoodNet model

        Args:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels

        """
        width1, width2, width3, width4 = width, int(width/2), int(width/4), int(width/8)
        height1, height2, height3, height4 = height, int(height/2), int(height/4), int(height/8)

        # conv1
        with tf.variable_scope('conv1') as scope:
            z_conv1 = self.conv_layer(self.images, 5, 3, 64)
        
        # pool1 下采样1
        with tf.variable_scope('pool1') as scope:
            h_pool1 = self.pool_layer(z_conv1)
        
        # conv01
        with tf.variable_scope('conv01') as scope:
            z_conv01 = self.conv_layer(h_pool1, 5, 64, 128)
        
        # pool2 下采样2
        with tf.variable_scope('pool2') as scope:
            h_pool2 = self.pool_layer(z_conv01)
        
        # conv02
        with tf.variable_scope('conv02') as scope:
            z_conv02 = self.conv_layer(h_pool2, 5, 128, 128)

        # deconv1 上采样1
        with tf.variable_scope('deconv1') as scope:
            deconv1 = self.deconv_layer(z_conv02, 5, batch_size, width2, height2, 128, 128)
        
        # conv12
        with tf.variable_scope('conv11') as scope:
            z_conv12 = self.conv_layer(deconv1, 5, 128, 64)

        # deconv2 上采样2
        with tf.variable_scope('deconv2') as scope:
            deconv2 = self.deconv_layer(z_conv12, 5, batch_size, width1, height1, 64, 64)
        
        # conv12
        with tf.variable_scope('conv12') as scope:
            z_conv12 = self.conv_layer(deconv2, 5, 64, 3)
        
        
        predict_images = z_conv12

        return predict_images

    # 网络2结构
    def u_net_model(self, batch_size, width, height):
        """Build th U-Net model

        Args:
            batch_size: value to batch train, value = 16,32,64,...abs
            width: value is image's width, value = 256,...abs
            height: value is image's height, value = 256,...abs
        Return:
            predict_images: Tensor, same with labels

        """
        
        predict_images = UNetLike.neural_networks_model(self.images, batch_size, width, height)
        return predict_images
