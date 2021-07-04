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
from sklearn.model_selection import KFold


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
    def __init__(self, data_path, data_mode, set_mode, valid_ratio = 0.2, index = None):
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

        # add k-fold
        if set_mode == 'train':
            self.imgs_path = self.imgs_path[:self.train_len]
        elif set_mode == 'valid':
            self.imgs_path = self.imgs_path[self.train_len:]
        elif set_mode == 'test':
            self.imgs_path = self.imgs_path[-1:]
        elif set_mode == 'kfold':
            self.imgs_path = np.array(self.imgs_path)
            self.imgs_path = self.imgs_path[index]
            self.imgs_path = self.imgs_path.tolist()

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

        return image, mask, label_path

    def __len__(self):
        return len(self.imgs_path)

class GAN_Data(Dataset):
    def __init__(self, data_path):
        self.simu_path = glob.glob(os.path.join(data_path,"aicm[1-9]/*/images/*.png"))
        self.simu_path += glob.glob(os.path.join(data_path,"aicm10/*/images/*.png"))

        self.intra_path = glob.glob(os.path.join(data_path,"aicm1[1-4]/*/images/*.png"))

        self.simu_len = len(self.simu_path)
        self.intra_len = len(self.intra_path)

        self.length_dataset = max(self.simu_len, self.intra_len) # 

        self.transform = None

        print('Finished reading the simu set of medical dataset ({} samples found)'
            .format(self.simu_len))

        print('Finished reading the intra set of medical dataset ({} samples found)'
            .format(self.intra_len))

    def __getitem__(self, index):
        simu_path = self.simu_path[index % self.simu_len]
        intra_path = self.intra_path[index % self.intra_len]

        simu_image = Image.open(simu_path).convert("RGB")
        intra_image = Image.open(intra_path).convert("RGB")

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        simu_image = transform(simu_image)
        intra_image = transform(intra_image)

        return simu_image, simu_path, intra_image

    def __len__(self):
        return self.length_dataset

if __name__ == "__main__":
    simulator_dataset = Medical_Data("./Traindata/","intra","train",valid_ratio = 0.0)
    # simulator_loader = torch.utils.data.DataLoader(dataset=simulator_dataset,
    #                                            batch_size=1, 
    #                                            shuffle=True)
    datalen = len(simulator_dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=123)
    data_idx = np.arange(datalen)
    print(data_idx)
    kfsplit = kf.split(data_idx)

    for fold, (train_idx, valid_idx) in enumerate(kfsplit):
        print("fold", fold)
        train_dataset = Medical_Data("./Traindata/","intra","kfold",valid_ratio = 0.0, index=train_idx)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        valid_dataset = Medical_Data("./Traindata/","intra","kfold",valid_ratio = 0.0, index=valid_idx)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=32, shuffle=True)

        dataiter = iter(valid_loader)
        images, labels, label_path = dataiter.next()
        print(label_path)
        print(images.shape)
        print(labels.shape)

    # image = im_convert(images, True)
    # label = im_convert(labels, False)
    # plt.imshow(image)
    # plt.savefig('./pic/intraimages.png')
    # plt.show()
    # plt.imshow(label)
    # plt.savefig('./pic/testheatmap.png')
    # plt.show()

    # GAN
    # gan_dataset = GAN_Data("./Traindata/")
    # gan_loader = torch.utils.data.DataLoader(dataset=gan_dataset,
    #                                            batch_size=1, 
    #                                            shuffle=True)
    # dataiter = iter(gan_loader)
    # simu, intra = dataiter.next()
    # print(simu.shape)
    # print(intra.shape)
    # simu = im_convert(simu, True)
    # intra = im_convert(intra, True)
    # plt.imshow(simu)
    # plt.savefig('./pic/simu.png')
    # plt.show()
    # plt.imshow(intra)
    # plt.savefig('./pic/intra.png')
    # plt.show()
