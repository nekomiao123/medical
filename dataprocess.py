import json
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.stats import multivariate_normal
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
        point_out = [point["x"],point["y"]]
        point_outs.append(point_out)
    '''
    change place
    '''
    circles = []
    # 画圆
    for a in point_outs:
        masks = generate_mask(height,width,6,a[0],a[1])
        xs,ys = np.where(masks == True)
        for i in range(len(xs)):
            circle = [xs[i],ys[i]]
            circles.append(circle)
    for b in circles:
        img[b[0],b[1]] = 1
    return img

def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image-mean))
    image = (image - mean)/np.sqrt(var)
    return image


def points_to_gaussian_heatmap(centers, height, width, scale):
    if centers:
        gaussians = []
        for x,y in centers:
            s = np.eye(2)*scale
            g = multivariate_normal(mean=(x,y), cov=s)
            gaussians.append(g)

        # create a grid of (x,y) coordinates at which to evaluate the kernels
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x,y)
        xxyy = np.stack([xx.ravel(), yy.ravel()]).T

        # evaluate kernels at grid points
        zz = sum(g.pdf(xxyy) for g in gaussians)
        img = zz.reshape((height,width))
    else:
        img = np.zeros((height,width))
    
    # img  = normalize(img)
    return img

def heatmap_generator(file_name, SCALE=64):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    point_outs = []
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    # img = np.zeros((height,width),dtype = np.uint8)
    for point in points:
        point_out = [point["x"],point["y"]]
        point_outs.append(point_out)

    img = points_to_gaussian_heatmap(point_outs, height, width, SCALE)
    return img

class Medical_Data(Dataset):
    def __init__(self, data_path, data_mode, set_mode, valid_ratio=0.2):
        '''
        data_path: data path.
        data_mode: simulator or intra data.
        set_mode:  train or valid or test.
        transform: for data augmentation
        '''
        self.data_path = data_path
        self.set_mode = set_mode
        self.transform = None
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
            self.imgs_path = self.imgs_path[-1:]

        print('Finished reading the {}_{} set of medical dataset ({} samples found)'
            .format(data_mode, set_mode, len(self.imgs_path)))

    def augment(self, image, code):
        print("data augment")

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("images","point_labels").replace(".png",".json")

        image = Image.open(image_path).convert("RGB")
        heatmap = np.array(heatmap_generator(label_path), dtype=np.float32)

        if self.set_mode == 'train':
            self.transform = transforms.Compose([
                # transforms.Resize((288,512)),
                # transforms.RandomResizedCrop(224,scale=(0.5,1.0)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif self.set_mode == 'valid':
           self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image = self.transform(image)
        heatmap = self.transform(heatmap)

        return image, heatmap, label_path

    def __len__(self):
        return len(self.imgs_path)

def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    return image

# if __name__ == "__main__":
#     # simulator_dataset = Medical_Data("./Traindata/","simulator","test")
#     simulator_dataset = Medical_Data("./Traindata/","simulator","train")
#     # simulator_dataset = Medical_Data("./Traindata/","simulator","valid")
#     # intra_dataset = Medical_Data("./Traindata/","intra","test")
#     simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
#                                                batch_size=1, 
#                                                shuffle=True)
#     dataiter = iter(simulator_loader)
#     images, labels = dataiter.next()
#     print(images.shape)
#     print(labels.shape)
#     image = im_convert(images, True)
#     label = im_convert(labels, False)
#     plt.imshow(image)
#     plt.savefig('./pic/images.png')
#     plt.show()
#     plt.imshow(label)
#     plt.savefig('./pic/heatmap.png')
#     plt.show()
    # print(images.shape)
    # print(labels.shape)
    # print(images)
    # print(labels)

class Medical_Data_test(Dataset):
    def __init__(self, data_path, data_mode, set_mode="test", valid_ratio=0.2):
        '''
        data_path: data path.
        data_mode: simulator or intra data.
        set_mode:  train or valid or test.
        transform: for data augmentation
        '''
        self.data_path = data_path
        self.set_mode = set_mode
        self.transform = None
        if data_mode == "simulator":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm[1-9]/*/images/*.png"))
            self.imgs_path += glob.glob(os.path.join(data_path,"aicm10/*/images/*.png"))
        elif data_mode == "intra":
            self.imgs_path = glob.glob(os.path.join(data_path,"aicm1[1-4]/*/images/*.png"))
        
        self.data_len = len(self.imgs_path)
        self.train_len = int(self.data_len * (1 - valid_ratio))
        self.imgs_path = self.imgs_path[-500:]

        print('Finished reading the {}_{} set of medical dataset ({} samples found)'
            .format(data_mode, set_mode, len(self.imgs_path)))

    def augment(self, image, code):
        print("data augment")

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("images","point_labels").replace(".png",".json")

        image = Image.open(image_path).convert("RGB")
        heatmap = np.array(heatmap_generator(label_path), dtype=np.float32)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = self.transform(image)
        heatmap = self.transform(heatmap)

        return image,heatmap,image_path

    def __len__(self):
        return len(self.imgs_path)