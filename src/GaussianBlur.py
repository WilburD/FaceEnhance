from PIL import Image, ImageFilter
import numpy as np
import math
import matplotlib.pyplot as plt
import time

# 对图像进行搞死模糊处理, RGB 3通道图像，未优化
# 输入image_array.shape = (width, height, channels), radius=半径
def gaussian_blur(image_array, radius):
    #1.权重矩阵
    weight_matrix = np.zeros((2*radius+1, 2*radius+1))
    for i in range(2*radius+1):
        for j in range(2*radius+1):
            weight_matrix[i, j] = GussianFunction2D(i-radius, j-radius)
    weight_matrix /= weight_matrix.sum()

    #2.根据权重矩阵weight_matrix更新新的图像像素值
    image_array_gau = np.zeros(image_array.shape)
    for i in range(image_array.shape[0]-2*radius):
        for j in range(image_array.shape[1]-2*radius):
            for k in range(3):
                # 时间完全耗在这一步运算上了
                a = np.multiply(weight_matrix, image_array[i:(i+2*radius+1), j:(j+2*radius+1), k])
                image_array_gau[i+radius, j+radius, k] = a.sum()
    return image_array_gau/255

# 优化后的高斯模糊函数, 用两次一维卷积代替二维卷积, 原因是高斯函数及其优美性, 但是速度未提升，可能是实验室电脑CPU太好
def gaussian_blur_opti(image_array, radius):
    #1.权重矩阵
    weight_matrix = np.zeros(2*radius+1)
    for i in range(2*radius+1):
        weight_matrix[i] = GaussianFunction1D(i-radius)
    weight_matrix /= weight_matrix.sum()

    #2.根据权重矩阵weight_matrix更新新的图像像素值
    #2.1 在y方向卷积
    image_array_gau_y = np.zeros(image_array.shape)
    for i in range(image_array.shape[0]-2*radius):
        for j in range(image_array.shape[1]-2*radius):
            for k in range(3):
                a = np.multiply(weight_matrix, image_array[i:(i+2*radius+1), j, k])
                image_array_gau_y[i+radius, j, k] = a.sum()

    #2.2 在x方向卷积
    image_array_gau = np.zeros(image_array.shape)
    for i in range(image_array.shape[0]-2*radius):
        for j in range(image_array.shape[1]-2*radius):
            for k in range(3):
                a = np.multiply(weight_matrix, image_array_gau_y[i, j:(j+2*radius+1), k])
                image_array_gau[i, j+radius, k] = a.sum()
    return image_array_gau/255

# 计算二维高斯函数的值G(x, y)
def GussianFunction2D(x, y):
    sigma = 2.5 #sigma为方差，sigma越大, 高斯函数图像越扁, 则模糊效果越明显; sigma越小, 趋向0的话, 则模糊效果越差越接近真实图像
    return (1/(2*math.pi*sigma**2)) * math.e**(-(x**2 + y**2)/(2*sigma**2))
# 计算一维高斯函数的值G(x)
def GaussianFunction1D(x):
    sigma = 2.5
    return (1/(((2*math.pi)**0.5)*sigma)) * math.e**(-(x**2)/(2*sigma**2))

rs_dir = '/home/wanglei/wl/face-enhance/resource'
image = Image.open(rs_dir + '/yangmi1.jpg')
# image = image.filter(ImageFilter.GaussianBlur(radius=3))
# image.show()
image_array = np.array(image)

time1 = time.time()
image_array_gau = gaussian_blur(image_array, 5)
time2 = time.time()
print('二维卷积耗时:' + str(time2-time1))

time1 = time.time()
image_array_gau = gaussian_blur_opti(image_array, 5)
time2 = time.time()
print('二次一维卷积耗时:' + str(time2-time1))

# plt.figure('GaussianBlur')
# plt.imshow(image_array_gau)
# plt.show()
