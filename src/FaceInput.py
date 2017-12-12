from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 得到训练集
def get_trains(start, end):
    outpath_degens = '/home/wanglei/wl/data/webface_degens'
    return get_images_array(outpath_degens, start, end)
def get_labels(start, end):
    outpath_labels = '/home/wanglei/wl/data/webface_labels'
    return get_images_array(outpath_labels, start, end)

# 根据路径获得图片像素数据
def get_images_array(path, start, end):
    images_array = np.empty([end-start, 256, 256, 3])
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