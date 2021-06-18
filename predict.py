import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models
from skimage import measure, draw, data, util
from skimage.filters import threshold_otsu, threshold_local,threshold_minimum,threshold_mean,rank
from skimage.morphology import disk
import skimage

from utils import im_convert, get_device
from dataprocess import Medical_Data_test, Medical_Data

# Specify the graphics card
torch.cuda.set_device(0)

device = get_device()

def to_json(points,img_path):
    '''
    生成json文件
    '''
    json_out ={}
    floder_name = img_path.split("/")[-3]
    video_name = img_path.split("/")[-2]
    image_name = img_path.split("/")[-1]
    json_out["folderName"] = floder_name
    json_out["subfolderName"] = video_name
    json_out["imageFileName"] = image_name

    json_out["points"] = points

    if not Path("./output/"+floder_name+"/"+video_name+"/").exists():
        os.makedirs(Path("./output/"+floder_name+"/"+video_name+"/"))
    
    with open("./output/"+floder_name+"/"+video_name+"/"+img_path.split("/")[-1].replace(".png",".json"),'w') as file_obj:
        json.dump(json_out,file_obj)

def OSTU(predict):
    radius = 2
    selem = disk(radius)
    threshold_global_otsu = threshold_otsu(predict)
    image_out = predict >= threshold_global_otsu
    # 开运算  圆形kernel
    kernel = skimage.morphology.disk(2)
    image_out =skimage.morphology.opening(image_out, kernel)
    # generate centre of mass
    label_img = measure.label(image_out, connectivity=2)
    props = measure.regionprops(label_img)
    # generate prediction points
    points = []
    for prop in props:
        # 这里注意x，y别搞反了,输入是 288x512,第零维度是y,第一维是x，
        point = {}
        point["y"] = prop.centroid[0]
        point["x"] = prop.centroid[1]
        points.append(point)

    return points


def check_out(image_out,points_big):
    for point_big in points_big:
        x = int(point_big["x"])
        y = int(point_big["y"])
        flag = 0
        for i in range(0,288):
            if image_out[i,x] == True and image_out[i-1,x] == False:
                image_out[i,x] = False
                flag =1
            if i<287 and image_out[i,x] == False and image_out[i+1,x]== False and flag==1:
                break
            elif i ==287:
                image_out[i,x] = False
    return image_out


def OSTU_test(predict):
    radius = 2
    selem = disk(radius)
    threshold_global_otsu = threshold_otsu(predict)
    image_out = predict >= threshold_global_otsu
    ## 画图
    # plt.imshow(image_out)
    # plt.savefig('./pic/image_out.png')
    # plt.show()

    # 开运算
    # 圆形kernel
    kernel = skimage.morphology.disk(2)
    image_out =skimage.morphology.opening(image_out, kernel)
    # print(image_out)
    # 画图
    # plt.imshow(image_out)
    # plt.savefig('./pic/open.png')

    # generate centre of mass
    # image_out = image_out[:,:,np.newaxis]
    label_img = measure.label(image_out, connectivity=2)
    props = measure.regionprops(label_img)
    # generate prediction points
    points = []
    areas = []
    for prop in props:
        # 这里注意x，y别搞反了,输入是 288x512,第零维度是y,第一维是x，
        point = {}
        point["y"] = prop.centroid[0]
        point["x"] = prop.centroid[1]
        points.append(point)
        areas.append(prop.area)
    print("before predict number",len(points))
    # for point in points:
    #     x = int(point["x"])
    #     y = int(point["y"])
    #     image_out[y,x] = 0
    # plt.imshow(image_out)
    # plt.savefig('./pic/open_points.png')
    # return points

    points_big = []
    area_average = int(sum(areas)/len(areas))
    for i in range(len(areas)):
        if areas[i] > area_average:
            points_big.append(points[i])
    if len(points_big):
        img_out = check_out(image_out,points_big)
        # print(points_big)
        # print("continue")
        label_img = measure.label(img_out, connectivity=2)
        props = measure.regionprops(label_img)
        points = []
        for prop in props:
        # 这里注意x，y别搞反了,输入是 288x512,第零维度是y,第一维是x，
            point = {}
            point["y"] = prop.centroid[0]
            point["x"] = prop.centroid[1]
            points.append(point)
        # print(len(points_big))
        print("after predict number",len(points))
        for point in points:
            x = int(point["x"])
            y = int(point["y"])
            image_out[y,x] = 0
        plt.imshow(image_out)
        plt.savefig('./pic/open_points.png')
        return points
    else:
        return points


def predict(model_path, test_loader):

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    for batch in tqdm(test_loader):
        imgs, labels, imgs_path = batch
        print(imgs_path)
        imgs = imgs.to(device)
        labels = labels.to(device)

        img = im_convert(imgs, True)
        plt.imshow(img)
        plt.savefig('./pic/original.png')
        plt.show()

        label = im_convert(labels, False)
        plt.imshow(label)
        plt.savefig('./pic/original_label.png')
        plt.show()

        with torch.no_grad():
            logits = torch.sigmoid(model(imgs))
            # logits = (logits > 1e-7)
        # print(logits)

        logit = im_convert(logits, False)
        plt.imshow(logit)
        plt.savefig('./pic/predict.png')
        plt.show()
        # print(imgs_path)
        # print(logit)
        points = OSTU(logit)
        points = OSTU_test(logit)
        # 生成json文件
        # to_json(points,img_path[0])
        # for i in range(len(imgs_path)):
        #     predict = im_convert(logits[i], False)
        #     OSTU(predict,imgs_path[i])

def main():
    batch_size = 1
    num_workers = 1
    test_path = './Traindata/'
    model_path = './model/new_Unet.pt'
    test_dataset = Medical_Data(test_path, data_mode='simulator', set_mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False
        )

    predict(model_path, test_loader)
    # print("Done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

if __name__ == "__main__":
    main()