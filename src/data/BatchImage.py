from PIL import Image, ImageFilter
import numpy as np
import os, random

# 将图片缩小成指定大小
def shrink_image(inputpath, outputpath, size):
    image = Image.open(inputpath)
    image = image.resize((size, size), Image.ANTIALIAS)
    image.save(outputpath)

# 讲图片存储到相应文件夹, 随机打乱
def save_image(inputpath, outpath_guides, outpath_degens, outpath_labels):
    os.mkdir(outpath_guides)
    os.mkdir(outpath_degens)
    os.mkdir(outpath_labels)

    list_name = []
    for parent, dirnames, filenames in os.walk(inputpath):
        for filename in filenames:
            list_name.append(filename)
    list_name = sorted(list_name)

    list_name_s = []
    for i in range(5000):
        list_name_s.append(list_name[i])
    random.shuffle(list_name_s)

    for i in range(5000):
        t = i + 1
        s = str(t).zfill(5)
        image = Image.open(inputpath + '/' + list_name_s[i])
        image_guide = image.crop((0, 0, 256, 256))
        image_label = image.crop((256, 0, 512, 256))
        image_degen = image_label.filter(ImageFilter.GaussianBlur(radius=3))

        image_guide.save(outpath_guides + '/guide_' + s + '.png')
        image_degen.save(outpath_degens + '/degen_' + s + '.png')
        image_label.save(outpath_labels + '/label_' + s + '.png')
        print(list_name_s[i], t)
            
# 缩小图像规模
# 256x256x3 -> 64x64x3
def downscale_image(inputpath, outpath, size):
    os.mkdir(outpath)
    list_name = []
    for parent, dirnames, filenames in os.walk(inputpath):
        for filename in filenames:
            list_name.append(filename)
    list_name = sorted(list_name)

    for i in range(len(list_name)):
        image = Image.open(inputpath + '/' + list_name[i])
        image = image.resize((size, size), Image.ANTIALIAS)
        image.save(outpath + '/' + list_name[i])
        print(outpath + '/' + list_name[i])

inputpath = '/home/wanglei/wl/data/WebFace'
outpath_guides = '/home/wanglei/wl/data/webface_guides'
outpath_degens = '/home/wanglei/wl/data/webface_degens'
outpath_labels = '/home/wanglei/wl/data/webface_labels'

# save_image(inputpath, outpath_guides, outpath_degens, outpath_labels)
# random_sort(outpath_guides)
# downscale_image(outpath_guides, '/home/wanglei/wl/data/webface_guides_64x64', 64)
# downscale_image(outpath_degens, '/home/wanglei/wl/data/webface_degens_64x64', 64)
# downscale_image(outpath_labels, '/home/wanglei/wl/data/webface_labels_64x64', 64)

# shrink_image('/home/wanglei/图片/test.jpg', 
#             '/home/wanglei/图片/test1.jpg', 
#             256)