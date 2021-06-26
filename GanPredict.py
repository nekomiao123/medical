import torch
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

from dataprocess import GAN_Data
from GanNetwork import Discriminator, Generator
from utils import get_device
from utils import save_checkpoint, load_checkpoint

# Specify the graphics card
torch.cuda.set_device(4)

test_path = './Traindata/'
train_name = '4gan'

default_config = dict(
    batch_size=1,
    num_epoch=40,
    learning_rate=1e-5,            # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=5,
    warm_up_epochs=5,

    SAVE_MODEL = False,
    LOAD_MODEL = True,
    LAMBDA_CYCLE = 10,
    LAMBDA_IDENTITY = 0.5,

    CHECKPOINT_GEN_SIMU = './ganmodel/gen_simu_'+ train_name + '.pth',
    CHECKPOINT_GEN_INTRA = './ganmodel/gen_intra_'+ train_name + '.pth',
    CHECKPOINT_DIS_SIMU = './ganmodel/dis_simu_'+ train_name + '.pth',
    CHECKPOINT_DIS_INTRA = './ganmodel/dis_intra_'+ train_name + '.pth'
)

config = default_config
device = get_device()

def pre_data(batch_size, num_workers=5):

    test_dataset = GAN_Data(test_path)

    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1, 
            shuffle=False,
            num_workers=num_workers
        )

    return test_loader

def predict(test_loader, learning_rate):
    gen_simu = Generator(img_channels=3, num_residuals=9).to(device)
    gen_intra = Generator(img_channels=3, num_residuals=9).to(device)

    optim_gen = optim.Adam(
        list(gen_simu.parameters()) + list(gen_intra.parameters()),
        lr = learning_rate
    )

    if config['LOAD_MODEL']:
        load_checkpoint(
            config['CHECKPOINT_GEN_SIMU'], gen_simu, optim_gen, learning_rate,
        )
        load_checkpoint(
            config['CHECKPOINT_GEN_INTRA'], gen_intra, optim_gen, learning_rate,
        )

    gen_intra.eval()

    dataiter = iter(test_loader)
    simu, simupath ,intra = dataiter.next()
    print(simupath)

    simu = simu.to(device)
    intra = intra.to(device)

    with torch.no_grad():
        fake_intra = gen_intra(simu)
        cylic_simu = gen_simu(fake_intra)

    save_image(fake_intra, "pic/"+train_name+"preintra.png")
    save_image(simu, "pic/"+train_name+"simu.png")
    save_image(cylic_simu, "pic/"+train_name+"cylic_simu.png")


def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epoch = config['num_epoch']

    test_loader = pre_data(batch_size, num_workers)
    predict(test_loader, learning_rate)

    

if __name__ == "__main__":
    main()
