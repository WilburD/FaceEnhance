# 本系统的深度学习神经网络模型
import tensorflow as tf
import numpy as np
import FaceInput
from model import GoodNet
import matplotlib.pyplot as plt

# 处理获取数据集
def inputs(start, end, image_size):
    imagedata = FaceInput.ImageData(start, end, image_size)
    images = imagedata.get_input_images()
    labels = imagedata.get_label_images()
    # images = FaceInput.get_trains(start, end, image_size)
    # labels = FaceInput.get_labels(start, end, image_size)
    return images, labels

# 网络模型
def neural_networks_model(xs, batch_size, image_size, model_type):
    if model_type == 'unet':
        goodnet = GoodNet.GoodNet(xs, 0)
        # model = goodnet.coarse_net_model(batch_size, image_size, image_size)
        # model = goodnet.fine_net_model(batch_size, image_size, image_size)
        # model = goodnet.u_net_2_model(batch_size, image_size, image_size)
    return model

# 代价函数, 欧式距离
def compute_loss(output_images, label_images):
    loss = tf.reduce_mean(tf.square(label_images - output_images))
    return loss

# 训练
def train(iters, batch_size, train_num, model_path, image_size):
    xs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    ys = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

    goodnet = GoodNet.GoodNet(xs, ys)
    coarse_images, coarse_parms = goodnet.coarse_net_model(batch_size, image_size, image_size)
    coarse_loss = goodnet.coarse_loss(coarse_images)

    fine_images, fine_parms = goodnet.fine_net_model(xs, batch_size, image_size, image_size)
    fine_loss = goodnet.fine_loss(fine_images)

    coarse_train_step = tf.train.GradientDescentOptimizer(0.1).minimize(coarse_loss, 
                                                                        var_list=coarse_parms)

    fine_train_step = tf.train.GradientDescentOptimizer(0.08).minimize(fine_loss, 
                                                                        var_list=fine_parms)

    saver = tf.train.Saver()
    file_log = open('log.txt', 'wt')
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)

        flag = 1
        for i in range(iters):
            print('--------------------------------------------------------------')
            file_log.write('--------------------------------------------------------------\n')

            for t in range(0, train_num, batch_size):
                xs_batch, ys_batch = inputs(t, t+batch_size, image_size)
                if flag == 1:
                    sess.run(coarse_train_step, feed_dict={xs:xs_batch, ys:ys_batch})
                else:
                    sess.run(fine_train_step, feed_dict={xs:xs_batch, ys:ys_batch})
                if t % 128 == 0:
                    cost1 = sess.run(coarse_loss, feed_dict={xs: xs_batch, ys: ys_batch})
                    # cost2 = sess.run(fine_loss, feed_dict={xs: xs_batch, ys: ys_batch})
                    cost2 = 1
                    print('iters:%s,batch:%s, loss1:%s,loss2:%s' % (i, t, cost1, cost2))
                    file_log.write('iters:%s,batch:%s, loss1:%s,loss2:%s \n' % (i, t, cost1, cost2))
            # if i % 200 == 0:
            #     xs_batch, ys_batch = inputs(0, batch_size, image_size)
            #     cost1 = sess.run(coarse_loss, feed_dict={xs: xs_batch, ys: ys_batch})
            #     cost2 = sess.run(fine_loss, feed_dict={xs: xs_batch, ys: ys_batch})
            #     print('************cost1:%s, cost2:%s **********' % (cost1, cost2))
            #     if cost1 < cost2:
            #         flag = 0
            #     else:
            #         flag = 1

            if i % 50 == 0:
                xs_batch, ys_batch = inputs(0, batch_size, image_size)
                predict_images1 = xs_batch[0]
                predict_images2 = sess.run(coarse_images[0], feed_dict={xs: xs_batch, ys: ys_batch})
                predict_images3 = sess.run(fine_images[0], feed_dict={xs: xs_batch, ys: ys_batch})
                predict_images4 = ys_batch[0]
                fig = plt.figure()
                ax = fig.add_subplot(2, 2, 1)
                ax.imshow(predict_images1)
                plt.axis('off')
                ax = fig.add_subplot(2, 2, 3)
                ax.imshow(predict_images2)
                plt.axis('off')
                ax = fig.add_subplot(2, 2, 4)
                ax.imshow(predict_images3)
                plt.axis('off')
                ax = fig.add_subplot(2, 2, 2)
                ax.imshow(predict_images4)
                plt.axis('off')
                plt.savefig('/home/wanglei/图片/' + str(i) + '.png')
                # plt.show()
                saver.save(sess, model_path)
                print('***********************保存成功***********************')
    
    # writer.close()
    sess.close()

# 预测
def predict(input_image, label_image, save_path, image_size, it, batch_size):
    goodnet = GoodNet.GoodNet(tf.cast(input_image ,tf.float32), 0)
    coarse_images, p1 = goodnet.coarse_net_model(batch_size, image_size, image_size)
    fine_images, p2 = goodnet.fine_net_model(coarse_images, batch_size, image_size, image_size)

    saver = tf.train.Saver()
    print(coarse_images)
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        predict_image1 = coarse_images.eval(session = sess)
        predict_image2 = fine_images.eval(session = sess)
        fig = plt.figure()
        num = batch_size + it
        for t in range(it, num):
            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(input_image[t - it])
            plt.axis('off')
            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(label_image[t - it])
            plt.axis('off')
            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(predict_image1[t - it])
            plt.axis('off')
            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(predict_image2[t - it])
            plt.axis('off')
            plt.savefig('/home/wanglei/图片/' + str(t) + '.png')
        # plt.show()

# 训练测试UNet model
def u_net_main():
    iters = 200000 # 迭代次数
    batch_size = 32
    train_num = 512 # 训练集数量
    image_size = 256
    model_path_unet = '/home/wanglei/wl/model/model_unet.ckpt' # UNet model 256x256

    # t = 100001
    # imagedata = FaceInput.ImageData(0, 1, 256)
    # x = imagedata.get_image_by_path('/home/wanglei/wl/face-enhance/resource/yangmi22.jpg')
    # y = imagedata.get_image_by_path('/home/wanglei/wl/face-enhance/resource/yangmi11.jpg')
    # x = imagedata.get_image_by_path('/home/wanglei/图片/test1.jpg')
    # y = imagedata.get_image_by_path('/home/wanglei/图片/test1.jpg')

    t = 0
    num = 10
    x, y = inputs(t, t+num, image_size)

    if image_size == 256:
        train(iters, batch_size, train_num, model_path_unet, image_size)
        # predict(x, y, model_path_unet, image_size, t, num)
    return 0

u_net_main()