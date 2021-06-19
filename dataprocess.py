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
from utils import im_convert
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
    _range = np.max(image) - np.min(image)
    img = (image - np.min(image)) / _range
    return img

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

        # normalize to 0 and 1
        img = normalize(img)
    else:
        img = np.zeros((height,width))

    return img

def heatmap_generator(file_name, SCALE = 32):
    file_in = json.load(open(file_name))
    points = file_in["points"]
    height = file_in["imageHeight"]
    width = file_in["imageWidth"]
    point_outs = []

    for point in points:
        point_out = [point["x"],point["y"]]
        point_outs.append(point_out)

    img = points_to_gaussian_heatmap(point_outs, height, width, SCALE)
    return img

class Medical_Data(Dataset):
    def __init__(self, data_path, data_mode, set_mode, valid_ratio = 0.2):
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

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace("images","point_labels").replace(".png",".json")

        image = np.array(Image.open(image_path), dtype=np.float32) / 255.
        mask = np.array(heatmap_generator(label_path))

        if self.set_mode == 'train':
            self.transform = A.Compose(
                    [
                        A.Resize(width=512, height=288),
                        A.Rotate(limit=40, p = 0.7, border_mode=cv2.BORDER_CONSTANT),
                        A.HorizontalFlip(p = 0.5),
                        A.VerticalFlip(p = 0.5),
                        A.ColorJitter(p = 0.5),
                        A.RandomBrightnessContrast(p=0.5),
                        # A.OneOf([
                        #     A.Blur(blur_limit=3, p=0.5),
                        #     A.ColorJitter(p=0.5),
                        # ], p=1.0),
                        # A.Normalize(
                        #     mean=[0, 0, 0],
                        #     std=[1, 1, 1],
                        #     max_pixel_value=255,
                        # ),
                        ToTensorV2(),
                    ]
                )
        elif self.set_mode == 'valid':
           self.transform = A.Compose(
                    [
                        A.Resize(width=512, height=288),
                        ToTensorV2(),
                    ]
                )
        else:
             self.transform = A.Compose(
                    [
                        A.Resize(width=512, height=288),
                        ToTensorV2(),
                    ]
                )

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            # 增加一个维度
            mask = mask.unsqueeze(0)

            # print(image)
            # print(mask)
            # print(image.dtype)
            # print(mask.dtype)
        return image, mask, label_path

    def __len__(self):
        return len(self.imgs_path)

if __name__ == "__main__":
    # simulator_dataset = Medical_Data("./Traindata/","simulator","test")
    simulator_dataset = Medical_Data("./Traindata/","simulator","train")
    # simulator_dataset = Medical_Data("./Traindata/","simulator","valid")
    # intra_dataset = Medical_Data("./Traindata/","intra","test")
    simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    dataiter = iter(simulator_loader)
    images, labels, label_path = dataiter.next()
    print(label_path)
    print(images.shape)
    print(labels.shape)
    image = im_convert(images, True)
    label = im_convert(labels, False)
    plt.imshow(image)
    plt.savefig('./pic/images.png')
    plt.show()
    plt.imshow(label)
    plt.savefig('./pic/testheatmap.png')
    plt.show()
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