from PIL import Image
import numpy as np
import os

res_dir = '/home/wanglei/wl/SampleScale4/TestSampleScale4'
res_dir1 = '/home/wanglei/wl/SampleScale4/WebFace'
i = 8608
for parent, dirnames, filenames in os.walk(res_dir):
    # print(parent)
    for filename in filenames:
        i = i + 1
        s = str(i).zfill(5)
        image = Image.open(res_dir + '/' + filename)
        image = image.crop((256, 0, 768, 256))
        image.save(res_dir1 + '/image_' + s + '.png')
        print(s)
        # print(image.size)
        # print(filename)
        