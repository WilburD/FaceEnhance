from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 处理图片数据集
class ImageData:
    def __init__(self, start, end, image_size):
        self.start = start
        self.end = end
        self.image_size = image_size
    
    # 训练输入集
    def get_input_images(self):
        return get_trains(self.start, self.end, self.image_size)
    
    # 训练标签集
    def get_label_images(self):
        return get_labels(self.start, self.end, self.image_size)

# start 初始图片序号
# end 结束图片序号
# image_size 图片大小，宽度和高度
def get_trains(start, end, image_size):
    if image_size == 256:
        outpath_degens = '/home/wanglei/wl/data/webface_degens'
    else:
        outpath_degens = '/home/wanglei/wl/data/webface_degens_64x64'
    return get_images_array(outpath_degens, start, end, image_size)
def get_labels(start, end, image_size):
    if image_size == 256:
        outpath_labels = '/home/wanglei/wl/data/webface_labels'
    else:
        outpath_labels = '/home/wanglei/wl/data/webface_labels_64x64'
    return get_images_array(outpath_labels, start, end, image_size)

# 根据路径获得图片像素数据
def get_images_array(path, start, end, image_size):
    images_array = np.empty([end-start, image_size, image_size, 3])
    list_name = []
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            list_name.append(filename)
    list_name = sorted(list_name)

    for i in range(start, end):
        image = Image.open(path + '/' + list_name[i])
        # image.show()
        images_array[i-start] = np.array(image)/255
        # images_array[i] = image
    return images_array

# 测试
def main():
    X = get_trains(0, 32)
    Y = get_labels(0, 32)
    print(X.shape)

    fig = plt.figure('test')
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(X[0])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(Y[0])
    # plt.axis('off')
    plt.show()

# main()