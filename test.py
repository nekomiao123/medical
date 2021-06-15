# # import glob
# # import os
# # import json
# import matplotlib.pyplot as plt
# from skimage import measure,data,color,io
# from skimage import data, util
# from skimage.measure import label, regionprops
# # def label2mask(file_name):
# #     file_in = json.load(open(file_name))
# #     points = file_in["points"]
# #     x = []
# #     if len(points):
# #         for point in points:
# #             if int(point["x"]) <= 6:
# #                 print("x太小了：",file_name)
# #             if int(point["x"]) >= 506:
# #                 print("x太大了",file_name)
# #             if int(point["y"]) <= 6:
# #                 print("y太小了：",file_name)
# #             if int(point["y"]) >= 282:
# #                 print("y太大了",file_name)
# #             x.append(int(point["x"]))
# #             # print(point["x"])
# #         return min(x),max(x)
# #     else:
# #         return 512,0
# #     # point_outs = []
# #     # height = file_in["imageHeight"]
# #     # width = file_in["imageWidth"]
# #     # img = np.zeros((height,width,3),dtype = np.uint8)
# #     # for point in points:
# #     #     point_out = []
# #     #     point_out.append(point["x"])
# #     #     point_out.append(point["y"])
# #     #     point_outs.append(point_out)
# #     # circles = []
# #     # # 画圆
# #     # for a in point_outs:
# #     #     masks = generate_mask(height,width,6,a[1],a[0])
# #     #     xs,ys = np.where(masks == True)
# #     #     for i in range(len(xs)):
# #     #         circle = []
# #     #         circle.append(xs[i])
# #     #         circle.append(ys[i])
# #     #         circles.append(circle)
# #     # for b in circles:
# #     #     img[b[1],b[0]] = [1,1,1]
# #     # return img

# # paths = glob.glob(os.path.join("./Traindata/","*/*/point_labels/*.json"))
# # # print(paths)
# # x_min=[]
# # x_max=[]
# # for p in paths:
# #     # print(p)
# #     a,b = label2mask(p)
# #     x_min.append(a)
# #     x_max.append(b)
# # print(min(x_min),max(x_max))
# image_out = io.imread("./pic/global_otsu.png")
# print(type(image_out))
# # print(image_out[:,:,1])
# # print(measure.find_contours(image_out[:,:,1],0.5))
# img = util.img_as_ubyte(image_out) > 110
# print(type(img),img.shape)
# label_img = measure.label(img, connectivity=img.ndim)
# props = measure.regionprops(label_img)
# print(props[0].centroid)
# # for prop in props:
# #     print(prop["centroid"])
from pathlib import Path
import os

if not Path("./output/aaa/").exists():
    os.makedirs(Path("./output/aaa/"))
