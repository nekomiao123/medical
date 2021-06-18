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
    ## 画图
    # plt.imshow(image_out)
    # plt.savefig('./pic/image_out.png')
    # plt.show()

    # 开运算
    # 圆形kernel
    kernel = skimage.morphology.disk(2)
    image_out =skimage.morphology.opening(image_out, kernel)
    # 画图
    # plt.imshow(image_out)
    # plt.savefig('./pic/open.png')

    # generate centre of mass
    # image_out = image_out[:,:,np.newaxis]
    label_img = measure.label(image_out, connectivity=2)
    props = measure.regionprops(label_img)
    # generate prediction points
    points = []
    for prop in props:
        # 这里注意x，y别搞反了,输入是 288x512,第零维度是y,第一维是x，
        point = {}
        point["x"] = prop.centroid[1]
        point["y"] = prop.centroid[0]
        points.append(point)

    # for point in points:
    #     x = int(point["x"])
    #     y = int(point["y"])
    #     image_out[y,x] = 0
    # plt.imshow(image_out)
    # plt.savefig('./pic/open_points.png')

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
        # points = OSTU(logit)
        # 生成json文件
        # to_json(points,img_path[0])
        # for i in range(len(imgs_path)):
        #     predict = im_convert(logits[i], False)
        #     to_OSTU(predict,imgs_path[i])

def main():
    batch_size = 1
    num_workers = 1
    test_path = './Traindata/'
    model_path = './model/nor_unet.pt'
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