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
from network import UNET
from utils import check_accuracy

# use this to record my loss
import wandb

# Specify the graphics card
torch.cuda.set_device(4)

# hyperparameter
default_config = dict(
    batch_size=16,
    num_epoch=200,
    learning_rate=3e-4,          # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=5,
    warm_up_epochs=10,
    model_path = './model/heatmap_test.pt'
)

wandb.init(project='Medical', entity='nekokiku', config=default_config, name='test')
config = wandb.config
# config = default_config
train_path = './Traindata/'

#check device
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

def pre_data(batch_size, num_workers):
    train_dataset = Medical_Data(train_path, data_mode='simulator', set_mode='train')
    val_dataset = Medical_Data(train_path, data_mode='simulator', set_mode='valid')
    # test_dataset = Medical_Data(test_path, data_mode, set_mode='test')
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )
    # test_loader = torch.utils.data.DataLoader(
    #         dataset=test_dataset,
    #         batch_size=batch_size, 
    #         shuffle=False,
    #         num_workers=num_workers
    #     )

    return train_loader, val_loader

def train(train_loader, val_loader, learning_rate, weight_decay, num_epoch, model_path):

    # model 
    model = my_unet()
    model = model.to(device)
    model.device = device

    # For the segmentation task, we use BCEWithLogitsLoss as the measurement of performance.
    criterion = nn.BCEWithLogitsLoss()

    # Initialize optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')

    for epoch in range(num_epoch):
        # ---------- Training ----------
        model.train() 
        train_loss = []
        for batch in tqdm(train_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = sum(train_loss) / len(train_loss)
        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}")
        # check_accuracy(train_loader, model)

        # ---------- Validation ----------
        model.eval()
        valid_loss = []

        for batch in tqdm(val_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(imgs)

            loss = criterion(logits, labels)

            valid_loss.append(loss.item())

        valid_loss = sum(valid_loss) / len(valid_loss)

        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}")
        check_accuracy(val_loader, model)

        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': valid_loss})

        # if the model improves, save a checkpoint at this epoch
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model, model_path)
            print('saving model with best_loss {:.5f}'.format(best_loss))


def main():
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epoch = config['num_epoch']
    model_path = config['model_path']
    train_loader, val_loader = pre_data(batch_size, num_workers)
    train(train_loader, val_loader, learning_rate, weight_decay, num_epoch, model_path)

if __name__ == "__main__":
    main()