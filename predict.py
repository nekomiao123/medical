# This is for the progress bar.
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
import torchvision.models as models

from dataprocess import Medical_Data
from network import my_unet


# Specify the graphics card
torch.cuda.set_device(4)

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

def predict(model_path, test_loader, saveFileName):

    model=torch.load(model_path)
    model = model.to(device)

    model.eval()


def main():
    test_path = ''
    test_dataset = Medical_Data(test_path, data_mode='simulator', set_mode='test')

    predict()


if __name__ == "__main__":
    main()