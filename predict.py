# This is for the progress bar.
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from dataprocess import Medical_Data
from network import my_unet
from skimage.filters import threshold_otsu, threshold_local,threshold_minimum,threshold_mean,rank
from skimage.morphology import disk

# Specify the graphics card
torch.cuda.set_device(4)

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()


def im_convert(tensor, ifimg):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if ifimg:
        image = image.transpose(1,2,0)
    # image = image.clip(0, 1)
    return image

def predict(model_path, test_loader):

    model=torch.load(model_path)
    model = model.to(device)
    model.eval()

    for batch in tqdm(test_loader):
        imgs, labels = batch
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

        predict = im_convert(logits, False)
        ## local
        radius = 15
        selem = disk(radius)
        local_otsu = rank.otsu(predict, selem)
        threshold_global_otsu = threshold_otsu(predict)
        global_otsu = predict >= threshold_global_otsu
        plt.imshow(global_otsu)
        plt.savefig('./pic/global_otsu.png')
        # plt.imshow(threshold_global_otsu)
        plt.savefig('./pic/threshold_global_otsu.png')
        plt.show()


def main():
    batch_size = 1
    num_workers = 2
    test_path = './Traindata/'
    model_path = './model/test2.pt'
    test_dataset = Medical_Data(test_path, data_mode='simulator', set_mode='test')
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    
    predict(model_path, test_loader)


if __name__ == "__main__":
    main()