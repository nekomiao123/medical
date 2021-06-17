from skimage import io
import numpy as np
import scipy.ndimage as ndi
from skimage import measure,color
import matplotlib.pyplot as plt

# # img_path = "./test.png"
# image_out=io.imread("test.png")
# # io.imshow(img)
# label_img = measure.label(image_out, connectivity=2)
# props = measure.regionprops(label_img)
# points = []
# for prop in props:
#     # x,y
#     point = {}
#     point["x"] = prop.centroid[0]
#     point["y"] = prop.centroid[1]
#     points.append(point)
# print(points)


def microstructure(l=256):
    n = 5
    x, y = np.ogrid[0:l, 0:l]  #生成网络
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)  #随机数种子
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n)) #高斯滤波
    return mask > mask.mean()
 
data = microstructure(l=128)*1 #生成测试图片
print(data)
labels=measure.label(data,connectivity=2) 
props = measure.regionprops(labels)
points = []
for prop in props:
    # x,y
    point = {}
    point["x"] = prop.centroid[0]
    point["y"] = prop.centroid[1]
    points.append(point)
print(points)
print(len(points))