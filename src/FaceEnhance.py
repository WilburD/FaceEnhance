# 本系统的深度学习神经网络模型
import tensorflow as tf
import numpy as np
import FaceInput, EncodeDecodeLike, UNetLike
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
        model = UNetLike.neural_networks_model(xs, batch_size, image_size)
    else:
        model = EncodeDecodeLike.neural_networks_model(xs, batch_size, image_size)
    return model

# 代价函数, 欧式距离
def compute_loss(output_images, label_images):
    loss = tf.reduce_mean(tf.square(label_images - output_images))
    return loss

# 训练
def train(iters, batch_size, train_num, model_path, image_size, model_type):
    xs = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    ys = tf.placeholder(tf.float32, [None, image_size, image_size, 3])

    output_images = neural_networks_model(xs, batch_size, image_size, model_type)
    cost_function = compute_loss(output_images, ys)
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost_function)
    # train_step = tf.train.AdamOptimizer(1e-4).minimize(cost_function)

    saver = tf.train.Saver()
    file_log = open('log.txt', 'wt')
    # loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, model_path)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        for i in range(iters):
            for t in range(0, train_num-batch_size, batch_size):
                xs_batch, ys_batch = inputs(t, t+batch_size, image_size)
                sess.run(train_step, feed_dict={xs:xs_batch, ys:ys_batch})
                if t % 4 == 0:
                    cost = sess.run(cost_function, feed_dict={xs: xs_batch, ys: ys_batch})
                    # loss.append(cost)
                    print('iters:%s, batch_add:%s, loss:%s' % (i, t, cost))
                    file_log.write('iters:%s, batch:%s, loss:%s \n' % (i, t, cost))
            file_log.write('--------------------------------------------------------------\n')
            print('--------------------------------------------------------------')
            if i % 2 == 0:
                saver.save(sess, model_path)
                print('*******************保存成功*******************')
    
    # writer.close()
    sess.close()

# 预测
def predict(input_image, label_image, save_path, image_size, model_type, it):
    output_image = neural_networks_model(tf.cast(input_image ,tf.float32), 1, image_size, model_type)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        predict_image = output_image.eval(session = sess)
        fig = plt.figure()
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(input_image[0])
        plt.axis('off')
        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(predict_image[0])
        plt.axis('off')
        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(label_image[0])
        plt.axis('off')
        plt.savefig('/home/wanglei/图片/' + str(it) + '.png')
        plt.show()

# 训练测试UNet model
def u_net_main():
    iters = 1000 # 迭代次数
    batch_size = 32
    train_num = 256 # 训练集数量
    image_size = 256
    model_type = 'unet'
    model_path_unet = '/home/wanglei/wl/model/model_unet.ckpt' # UNet model 256x256
    model_path_unet_64x64 = '/home/wanglei/wl/model/model_unet_64x64.ckpt' # UNet model 64x64

    # t = 100001
    # imagedata = FaceInput.ImageData(0, 1, 256)
    # x = imagedata.get_image_by_path('/home/wanglei/wl/face-enhance/resource/yangmi22.jpg')
    # y = imagedata.get_image_by_path('/home/wanglei/wl/face-enhance/resource/yangmi11.jpg')

    t = 302
    x, y = inputs(t, t+1, image_size)

    if image_size == 256:
        train(iters, batch_size, train_num, model_path_unet, image_size, model_type)
        # predict(x, y, model_path_unet_64x64, image_size, model_type, t)
    else : # 64x64 
        # train(iters, batch_size, train_num, model_path_unet_64x64, image_size, model_type)
        predict(x, y, model_path_unet_64x64, image_size, model_type, t)
    

# 训练测试EncodeDecode model
def encode_decode_main():
    iters = 5000 # 迭代次数
    batch_size = 32 
    train_num = 5000 # 训练集数量
    image_size = 64 # 图片大小
    model_type = 'endecode'
    model_path_endecode = '/home/wanglei/wl/model/model_endecode.ckpt' # EncodeDecode model 256x256
    model_path_endecode_64x64 = '/home/wanglei/wl/model/model_endecode_64x64.ckpt' # EncodeDecode model 64x64
    x, y = inputs(0, 1, image_size)
    if image_size == 256: # 256x256
        # train(iters, batch_size, train_num, model_path_path_endecode, image_size, model_type)
        predict(x, y, model_path_endecode_64x64, image_size, model_type, t)
    else : # 64x64 
        # train(iters, batch_size, train_num, model_path_endecode_64x64, image_size, model_type)
        predict(x, y, model_path_endecode_64x64, image_size, model_type, t)

u_net_main()
# encode_decode_main()