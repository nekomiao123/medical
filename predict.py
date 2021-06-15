# This is for the progress bar.
from dataprocess import Medical_Data_test, Medical_Data
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
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
import numpy as np
import json
from pathlib import Path
from shutil import copyfile

# Specify the graphics card
torch.cuda.set_device(0)

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

def to_json(props,img_path):
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

    points = []
    for prop in props:
        # x,y
        point = {}
        point["x"] = prop.centroid[0]
        point["y"] = prop.centroid[1]
        points.append(point)
    json_out["points"] = points

    if not Path("./output/"+floder_name+"/"+video_name+"/").exists():
        os.makedirs(Path("./output/"+floder_name+"/"+video_name+"/"))
    
    with open("./output/"+floder_name+"/"+video_name+"/"+img_path.split("/")[-1].replace(".png",".json"),'w') as file_obj:
        json.dump(json_out,file_obj)
        # print("This json file has been saved!")
    # copyfile(label_path, "./output/labels/"+label_path.split("/")[-1])

def to_OSTU(predict, img_path):
    radius = 2
    selem = disk(radius)
    local_otsu = rank.otsu(predict, selem)
    threshold_global_otsu = threshold_otsu(predict)
    image_out = predict >= threshold_global_otsu
    # plt.imshow(image_out)
    # plt.savefig('./pic/'+img_path.split("/")[-1].replace(".json","png"))
    # plt.show()

    # generate centre of mass
    image_out = image_out[:,:,np.newaxis]
    label_img = measure.label(image_out, connectivity=image_out.ndim)
    props = measure.regionprops(label_img)
    # 生成json文件
    to_json(props,img_path)

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # print(image.dtype)
    # image = image.astype(np.uint8)
    if ifimg:
        image = image.transpose(1,2,0)
    # image = image.clip(0, 1)
    return image

def predict(model_path, test_loader):

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    for batch in tqdm(test_loader):
        imgs, labels, imgs_path = batch
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
            # logits = (logits > 0.5).float()
        # print(logits)

        logit = im_convert(logits, False)
        plt.imshow(logit)
        plt.savefig('./pic/predict.png')
        plt.show()

        # for i in range(len(imgs_path)):
        #     predict = im_convert(logits[i], False)
        #     to_OSTU(predict,imgs_path[i])

def main():
    batch_size = 1
    num_workers = 1
    test_path = './Traindata/'
    model_path = './model/test2.pt'
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