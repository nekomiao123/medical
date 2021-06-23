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

# # Specify the graphics card
# torch.cuda.set_device(4)

# 使用多GPU保存模型的时候记得加上.module
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_path = './Traindata/'
train_name = 'gan'

default_config = dict(
    batch_size=8,
    num_epoch=10,
    learning_rate=1e-5,            # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=5,
    warm_up_epochs=5,

    SAVE_MODEL = True,
    LOAD_MODEL = False,
    LAMBDA_CYCLE = 10,
    LAMBDA_IDENTITY = 0.0,

    CHECKPOINT_GEN_SIMU = './ganmodel/gen_simu_'+ train_name + '.pth',
    CHECKPOINT_GEN_INTRA = './ganmodel/gen_intra_'+ train_name + '.pth',
    CHECKPOINT_DIS_SIMU = './ganmodel/dis_simu_'+ train_name + '.pth',
    CHECKPOINT_DIS_INTRA = './ganmodel/dis_intra_'+ train_name + '.pth'
)

config = default_config
device = get_device()

def pre_data(batch_size, num_workers=5):
    train_dataset = GAN_Data(train_path)
    # val_dataset = GAN_Data(train_path)
    val_loader = 1
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers
        )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False
    # )

    return train_loader, val_loader

def train(train_loader, val_loader, learning_rate, num_epoch):

    dis_simu = Discriminator(in_channels=3).to(device)
    dis_simu = nn.DataParallel(dis_simu, device_ids=gpus, output_device=gpus[0])
    dis_intra = Discriminator(in_channels=3).to(device)
    dis_intra = nn.DataParallel(dis_intra, device_ids=gpus, output_device=gpus[0])

    gen_simu = Generator(img_channels=3, num_residuals=9).to(device)
    gen_simu = nn.DataParallel(gen_simu, device_ids=gpus, output_device=gpus[0])
    gen_intra = Generator(img_channels=3, num_residuals=9).to(device)
    gen_intra = nn.DataParallel(gen_intra, device_ids=gpus, output_device=gpus[0])

    optim_dis = optim.Adam(
        list(dis_simu.parameters()) + list(dis_intra.parameters()),
        lr = learning_rate
    )
    optim_gen = optim.Adam(
        list(gen_simu.parameters()) + list(gen_intra.parameters()),
        lr = learning_rate
    )

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config['LOAD_MODEL']:
        load_checkpoint(
            config['CHECKPOINT_GEN_SIMU'], gen_simu, optim_gen, learning_rate,
        )
        load_checkpoint(
            config['CHECKPOINT_GEN_INTRA'], gen_intra, optim_gen, learning_rate,
        )
        load_checkpoint(
            config['CHECKPOINT_DIS_SIMU'], dis_simu, optim_dis, learning_rate,
        )
        load_checkpoint(
            config['CHECKPOINT_DIS_INTRA'], dis_intra, optim_dis, learning_rate,
        )

    for epoch in range(num_epoch):
        intra_reals = 0
        intra_fakes = 0
        loop = tqdm(train_loader, leave=True)

        for idx, (simu, intra) in enumerate(loop):
            simu = simu.to(device)
            intra = intra.to(device)

            # Train Discriminators simu and intra
            fake_intra = gen_intra(simu)
            D_intra_real = dis_intra(intra)
            D_intra_fake = dis_intra(fake_intra.detach())
            intra_reals += D_intra_real.mean().item()
            intra_fakes += D_intra_fake.mean().item()
            D_intra_real_loss = mse(D_intra_real, torch.ones_like(D_intra_real))
            D_intra_fake_loss = mse(D_intra_fake, torch.zeros_like(D_intra_fake))
            D_intra_loss = D_intra_real_loss + D_intra_fake_loss

            fake_simu = gen_simu(intra)
            D_simu_real = dis_simu(simu)
            D_simu_fake = dis_simu(fake_simu.detach())
            D_simu_real_loss = mse(D_simu_real, torch.ones_like(D_simu_real))
            D_simu_fake_loss = mse(D_simu_fake, torch.zeros_like(D_simu_fake))
            D_simu_loss = D_simu_real_loss + D_simu_fake_loss

            # put it togethor
            D_loss = (D_intra_loss + D_simu_loss)/2

            optim_dis.zero_grad()
            D_loss.backward()
            optim_dis.step()

            # Train Generators intra and simu
            # adversarial loss for both generators
            D_intra_fake = dis_intra(fake_intra)
            D_simu_fake = dis_simu(fake_simu)
            loss_G_intra = mse(D_intra_fake, torch.ones_like(D_intra_fake))
            loss_G_simu = mse(D_simu_fake, torch.ones_like(D_simu_fake))

            # cycle loss
            cycle_simu = gen_simu(fake_intra)
            cycle_intra = gen_intra(fake_simu)
            cycle_simu_loss = l1(simu, cycle_simu)
            cycle_intra_loss = l1(intra, cycle_intra)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_simu = gen_simu(simu)
            identity_intra = gen_intra(intra)
            identity_simu_loss = l1(simu, identity_simu)
            identity_intra_loss = l1(intra, identity_intra)

            G_loss = (
                loss_G_simu
                + loss_G_intra
                + cycle_simu_loss * config['LAMBDA_CYCLE']
                + cycle_intra_loss * config['LAMBDA_CYCLE']
                + identity_intra_loss * config['LAMBDA_IDENTITY']
                + identity_simu_loss * config['LAMBDA_IDENTITY']
            )

            optim_gen.zero_grad()
            G_loss.backward()
            optim_gen.step()

            if idx % 300 == 0:
                save_image(fake_intra*0.5+0.5, f"pic/intra_{idx}.png")
                save_image(fake_simu*0.5+0.5, f"pic/simu_{idx}.png")

            loop.set_postfix(intra_real=intra_reals/(idx+1), intra_fake=intra_fakes/(idx+1))

        if config['SAVE_MODEL']:
            save_checkpoint(gen_simu, optim_gen, filename=config['CHECKPOINT_GEN_SIMU'])
            save_checkpoint(gen_intra, optim_gen, filename=config['CHECKPOINT_GEN_INTRA'])
            save_checkpoint(dis_simu, optim_dis, filename=config['CHECKPOINT_DIS_SIMU'])
            save_checkpoint(dis_intra, optim_dis, filename=config['CHECKPOINT_DIS_INTRA'])

def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epoch = config['num_epoch']

    train_loader, val_loader = pre_data(batch_size, num_workers)
    train(train_loader, val_loader, learning_rate, num_epoch)
    

if __name__ == "__main__":
    main()
