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

        predict_images, fine_parms = UNetLike.neural_networks_model(coarse_images, batch_size, width, height)

        return predict_images, fine_parms

    # Fine loss function
    def fine_loss(self, fine_images):
        """compute the Fine Stage loss

        Args:
            fine_images: Tensor, the coarse stage predict_images
        Returns:
            fine_loss: value is coares stage loss
            
        """

        fine_loss1 = tf.reduce_mean(tf.square(self.labels[:, 100:200, 120:200] - fine_images[:, 100:200, 120:200]))
        fine_loss2 = tf.reduce_mean(tf.square(self.labels - fine_images))
        return (fine_loss1 + fine_loss2)/2

    # 子网络结构
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
        width1, width2, width3, width4 = width, int(width/2), int(width/4), int(width/8)
        height1, height2, height3, height4 = height, int(height/2), int(height/4), int(height/8)

        u_net_2_parms = []

        # conv1
        with tf.variable_scope('u_net2_conv1') as scope:
            z_conv1, weights, biases = self.conv_layer(self.images, 5, 3, 64)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)

        # pool1 下采样1
        with tf.variable_scope('u_net2_pool1') as scope:
            h_pool1 = self.pool_layer(z_conv1)
        
        # conv01
        with tf.variable_scope('u_net2_conv01') as scope:
            z_conv01, weights, biases = self.conv_layer(h_pool1, 5, 64, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        
        # pool2 下采样2
        with tf.variable_scope('u_net2_pool2') as scope:
            h_pool2 = self.pool_layer(z_conv01)
        
        # conv02
        with tf.variable_scope('u_net2_conv02') as scope:
            z_conv02, weights, biases = self.conv_layer(h_pool2, 5, 128, 256)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        
        # conv2
        with tf.variable_scope('u_net2_conv2') as scope:
            z_conv2, weights4, biases4  = self.conv_layer(z_conv02, 5, 256, 256)

        # deconv1 上采样1
        with tf.variable_scope('u_net2_deconv1') as scope:
            deconv1, kfilters = self.deconv_layer(z_conv02, 5, batch_size, width2, height2, 256, 128)
            u_net_2_parms.append(kfilters)

        # conv11
        with tf.variable_scope('u_net2_conv11') as scope:
            inputs = tf.concat([z_conv01, deconv1], axis=3)
            z_conv11, weights, biases = self.conv_layer(inputs, 5, 256, 128)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)

        # deconv2 上采样2
        with tf.variable_scope('u_net2_deconv2') as scope:
            deconv2, kfilters = self.deconv_layer(z_conv11, 5, batch_size, width1, height1, 128, 64)
            u_net_2_parms.append(kfilters)
            
        # conv12
        with tf.variable_scope('u_net2_conv12') as scope:
            inputs = tf.concat([z_conv1, deconv2], axis=3)
            z_conv12, weights, biases = self.conv_layer(inputs, 5, 128, 3)
            u_net_2_parms.append(weights)
            u_net_2_parms.append(biases)
        
        predict_images = z_conv12

        return predict_images, u_net_2_parms
