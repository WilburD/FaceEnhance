from PIL import Image, ImageFilter
import numpy as np
import os

def save_image(Y, inputpath, outpath, iter):
    i = (iter-1) * 1000
    for parent, dirnames, filenames in os.walk(inputpath):
        # print(parent)
        for filename in filenames:
            if i < iter * 1000 and i >= (iter-1) * 1000:
                image = Image.open(inputpath + '/' + filename)
                image = image.crop((256, 0, 512, 256))
                image_array = np.array(image)
                print('写入文件成功:', i)
                Y[i] = image_array
                i = i+1
            else:
                break
    np.save(outpath, Y)

# 讲图片存储到相应文件夹
def save_image(inputpath, outpath_guides, outpath_degens, outpath_labels):
    i = 0;
    for parent, dirnames, filenames in os.walk(inputpath):
        for filename in filenames:
            i += 1
            s = str(i).zfill(5)
            image = Image.open(inputpath + '/' + filename)
            image_guide = image.crop((0, 0, 256, 256))
            image_label = image.crop((256, 0, 512, 256))
            image_degen = image_label.filter(ImageFilter.GaussianBlur(radius=3))

            image_guide.save(outpath_guides + '/guide_' + s + '.png')
            image_degen.save(outpath_degens + '/degen_' + s + '.png')
            image_label.save(outpath_labels + '/label_' + s + '.png')
            print(filename, i)
            
# 缩小图像规模
# 256x256x3 -> 64x64x3
def downscale_image(inputpath, outpath, size):
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
# downscale_image(outpath_guides, '/home/wanglei/wl/data/webface_guides_64x64', 64)
# downscale_image(outpath_degens, '/home/wanglei/wl/data/webface_degens_64x64', 64)
# downscale_image(outpath_labels, '/home/wanglei/wl/data/webface_labels_64x64', 64)