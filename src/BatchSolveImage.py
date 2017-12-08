from PIL import Image
import numpy as np
import os

res_dir = '/home/wanglei/wl/data/WebFace'
for parent, dirnames, filenames in os.walk(res_dir):
    # print(parent)
    for filename in filenames:
        image = Image.open(res_dir + '/' + filename)
        # image = image.crop((256, 0, 768, 256))
        image.save(res_dir1 + '/image_' + s + '.png')
        