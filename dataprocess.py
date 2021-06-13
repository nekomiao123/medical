import json
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

def generate_mask(img_height,img_width,radius,center_x,center_y):
    y,x=np.ogrid[0:img_height,0:img_width]
    # circle mask
    mask = (x-center_x)**2+(y-center_y)**2<=radius**2
    return mask

def label2mask(file_name):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    point_outs = []
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    img = np.zeros((height,width),dtype = np.uint8)
    for point in points:
        point_out = []
        point_out.append(point["x"])
        point_out.append(point["y"])
        point_outs.append(point_out)
    # circles = []
    # # 画圆
    # for a in point_outs:
    #     masks = generate_mask(height,width,1,a[1],a[0])
    #     xs,ys = np.where(masks == True)
    #     for i in range(len(xs)):
    #         circle = []
    #         circle.append(xs[i])
    #         circle.append(ys[i])
    #         circles.append(circle)
    for b in point_outs:
        img[b[1],b[0]] = 255
    return img

class Medical_Data(Dataset):
    def __init__(self, data_path, data_mode, set_mode, valid_ratio=0.2):
        '''
        data_path: data path
        data_mode: simulator or intra data
        set_mode:  train or valid or test
        '''

        self.data_path = data_path
        if data_mode == "simulator":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm[1-9]/*/images/*.png"))
            self.imgs_path += glob.glob(os.path.join(data_path,"aicm10/*/images/*.png"))
        elif data_mode == "intra":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm1[1-4]/*/images/*.png"))
        
        self.data_len = len(self.imgs_path)
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if set_mode == 'train':
            self.imgs_path = self.imgs_path[:self.train_len]
        elif set_mode == 'valid':
            self.imgs_path = self.imgs_path[self.train_len:]
        elif set_mode == 'test':
            pass

        print('Finished reading the {}_{} set of medical dataset ({} samples found)'
            .format(data_mode, set_mode, len(self.imgs_path)))

    def augment(self, image, code):
        print("data augment")

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("images","point_labels").replace(".png",".json")
        img = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # mean，标准差
        ])

        image = transform(img)
        l = Image.fromarray(np.uint8(label2mask(label_path)))
        label = transform(l)

        # image = torch.from_numpy(np.array(img)) 
        # image = image.permute(2, 0, 1)
        # label = torch.from_numpy(label2mask(label_path))

        return image,label

    def __len__(self):
        return len(self.imgs_path)


def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    return image


if __name__ == "__main__":
    simulator_dataset = Medical_Data("./Traindata/","simulator","test")
    simulator_dataset = Medical_Data("./Traindata/","simulator","train")
    simulator_dataset = Medical_Data("./Traindata/","simulator","valid")
    # intra_dataset = Medical_Data("./Traindata/","intra","test")
    simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    dataiter = iter(simulator_loader)
    images, labels = dataiter.next()
    # print(images.shape)
    # print(labels.shape)
    # print(images)
    # print(labels)

