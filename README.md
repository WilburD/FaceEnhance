# face-enhance
本科毕业设计-基于深度学习的模糊人脸图像增强系统的设计与实现

## 简介
本系统目的是使模糊的人脸图片变清晰, 核心问题是去模糊，与图像超分辨率和图像修复问题不同。图像超分低分辨率图像与高分辨率图像本身差别不大，只有细微的极小局部的差别，而图像修复则是图像中缺失一部分，需要复原成原图，除去缺失部分其余部分和原图一模一样。总之，模糊图像增强是一个全局的图像超分问题和局部的图像修复的结合问题，核心是图像去模糊。

## 网络结构
基于Coarse-to-Fine结构，具体网络结构，略。
#### Coarse Stage

#### Fine Stage


## 测试
#### 测试1
coarse_loss = 0.0007, fine_loss = 0.0010
![image](https://github.com/wangleihitcs/face-enhance/raw/master/resource/test1.pn极