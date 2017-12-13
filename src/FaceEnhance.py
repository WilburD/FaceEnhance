# 本系统的深度学习神经网络模型
import tensorflow as tf
import numpy as np
import FaceInput, EncodeDecodeLike, UNetLike
import matplotlib.pyplot as plt

# 处理获取数据集
def inputs(start, end):
    images = FaceInput.get_trains(start, end)
    labels = FaceInput.get_labels(start, end)
    # images = tf.cast(images, tf.float32)
    # labels = tf.cast(labels, tf.float32)
    return images, labels

# 网络模型
def neural_networks_model(xs, batch_size, image_size):
    model = UNetLike.neural_networks_model(xs, batch_size, image_size)
    # model = EncodeDecodeLike.neural_networks_model(xs, batch_size)
    return model

# 代价函数
def compute_loss(output_images, label_images):
    loss = tf.reduce_mean(tf.square(label_images - output_images))
    return loss

# 训练
def train(iters, batch_size, train_num, model_path):
    image_size = 64
    xs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    ys = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

    # output_images = EncodeDecodeLike.neural_networks_model(xs, batch_size)
    output_images = neural_networks_model(xs, batch_size, image_size)
    cost_function = compute_loss(output_images, ys)
    train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cost_function)

    saver = tf.train.Saver()
    file_log = open('log.txt', 'wt')
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        for i in range(iters):
            for t in range(0, train_num-batch_size, batch_size):
                xs_batch, ys_batch = inputs(t, t+batch_size)
                sess.run(train_step, feed_dict={xs:xs_batch, ys:ys_batch})
                if t % 32 == 0:
                    cost = sess.run(cost_function, feed_dict={xs: xs_batch, ys: ys_batch})
                    print('iters:%s, batch_add:%s, loss:%s' % (i, t, cost))
                    file_log.write('iters:%s, batch_add:%s, loss:%s \n' % (i, t, cost))
            if i % 200 == 0:
                saver.save(sess, model_path)
    
    # writer.close()
    sess.close()

def predict(input_image, label_image, save_path):
    output_image = neural_networks_model(tf.cast(input_image ,tf.float32), 1, 64)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        predict_image = output_image.eval(session = sess)
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(input_image[0])
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(predict_image[0])
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(label_image[0])
        plt.show()

iters = 10000
batch_size = 32
train_num = 5000
model_path = '/home/wanglei/wl/model/model1.ckpt'
train(iters, batch_size, train_num, model_path)
# x, y = inputs(0, 1)
# predict(x, y, model_path)