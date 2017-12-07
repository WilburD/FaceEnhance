from PIL import Image, ImageFilter

rs_dir = '/home/wanglei/wl/face-enhance/resource'
image = Image.open(rs_dir + '/yangmi1.jpg')
# image = image.filter(ImageFilter.GaussianBlur(radius=3))
# image.show()
print(image.size)
