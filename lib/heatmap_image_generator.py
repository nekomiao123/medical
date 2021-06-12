import json
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def label2mask(file_name):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    point_outs = []
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    img = np.zeros((height,width,3),dtype = np.uint8)
    for point in points:
        point_out = []
        point_out.append(point["x"])
        point_out.append(point["y"])
        point_outs.append(point_out)
    for a in point_outs:
        img[a[0],a[1]] = [255,255,255]
    img_show = Image.fromarray(img,"RGB")
    img_show.show()
    img_show.save("./my.png")

def generate_heatmap_target(heatmap_size, landmarks, sigmas, scale=1.0, normalize=False):
    x_dim, y_dim = heatmap_size
    heatmap = np.zeros((x_dim, y_dim), dtype=float)
    