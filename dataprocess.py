import json
import glob
import os
import cv2
import numpy as np

def json_read(file_name):
    file_in = json.load(open(file_name))
    point_outs = []
    points = file_in["points"]
    for point in points:
        point_out = []
        point_out.append(point["x"])
        point_out.append(point["y"])
        point_outs.append(point_out)
    return point_outs

def label2mask(file_name):
    file_in = json.load(open(file_name))
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    img = []
    w = []
    for x in range(width):
        w.append(0)
    for y in range(height):
        img.append(w)
    point_outs = []
    points = file_in["points"]
    for point in points:
        point_out = []
        point_out.append(point["x"])
        point_out.append(point["y"])
        point_outs.append(point_out)
    for point in point_outs:
        img[point[0]][point[1]] = 255
    cv2.imshow('result.jpg',np.array(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

print(label2mask("./mycode/Traindata/aicm2/VID000_0/point_labels/sim_000000.json"))


# class Medical_Data(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.imgs_path = glob.glob(os.path.join(data_path,"image/*.png"))
#     def __getitem__(self, index):
#         image_path = self.imgs_path[index]
#         label_path = image_path.replace("images","point_labels")
#         image = cv2.imread(image_path)
#         label = json_read(label_path)