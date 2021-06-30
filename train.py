import math
import wandb
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
from evaluation import evaluate
from utils import get_device

# 使用多GPU保存模型的时候记得加上.module
gpus = [4, 5]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))

train_name = 'intra_test1'
# hyperparameter
default_config = dict(
    batch_size=32,
    num_epoch=200,
    learning_rate=1e-4,            # learning rate of Adam
    weight_decay=0.01,             # weight decay 
    num_workers=5,
    warm_up_epochs=5,
    model_path = './model/'+train_name+'.pt'
)

wandb.init(project='Medical', entity='nekokiku', config=default_config, name=train_name)
config = wandb.config
# config = default_config
train_path = './Traindata/'

device = get_device()

def pre_data(batch_size, num_workers, data_mode='intra'):
    train_dataset = Medical_Data(train_path, data_mode=data_mode, set_mode='train')
    val_dataset = Medical_Data(train_path, data_mode=data_mode, set_mode='valid')
    # test_dataset = Medical_Data(test_path, data_mode, set_mode='test')
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size, 
            shuffle=True,
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

# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        return 1 - dice

# Dice loss and BCE loss 
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

# Focal loss
ALPHA = 0.8
GAMMA = 2
class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE

        return focal_loss

def train(train_loader, val_loader, learning_rate, weight_decay, num_epoch, model_path):

    # model 
    model = my_unet(modelname='ResnextUnet')
    model = model.to(device)
    model.device = device

    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    criterion = DiceLoss()

    # Initialize optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # warm_up_with_cosine_lr
    warm_up_with_cosine_lr = lambda epoch: epoch / config['warm_up_epochs'] if epoch <= config['warm_up_epochs'] else 0.5 * ( math.cos((epoch - config['warm_up_epochs']) /(num_epoch - config['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)

    best_loss = float('inf')
    # best_dice = 0
    best_f1 = 0

    for epoch in range(num_epoch):
        sensitivity = 0
        precision = 0
        f1_score = 0
        true_positive_all_files = 0
        false_positive_all_files = 0
        false_negative_all_files = 0

        # ---------- Training ----------
        model.train() 
        train_loss = []
        for batch in tqdm(train_loader):
            imgs, labels, _= batch
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
            imgs, labels, label_path = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(imgs)

            true_positive_a_batch, false_positive_a_batch, false_negative_a_batch = evaluate(torch.sigmoid(logits), label_path)
            true_positive_all_files += true_positive_a_batch
            false_positive_all_files += false_positive_a_batch
            false_negative_all_files += false_negative_a_batch

            loss = criterion(logits, labels)
            valid_loss.append(loss.item())

        # sensitivity (SEN) = TP + P
        sensitivity = true_positive_all_files / \
            (true_positive_all_files + false_negative_all_files)
        # precision (PPV) = TP / PP
        precision = true_positive_all_files / \
            (true_positive_all_files + false_positive_all_files)
        # F1 score = (2 * PPV * SEN) / (PPV + SEN)
        eps = 1e-8
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity + eps)

        dice = check_accuracy(val_loader, model)
        valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f} precision = {precision:.5f} sensitivity = {sensitivity:.5f} f1_score = {f1_score:.5f} dice = {dice:.5f}")

        # learning rate decay and print 
        scheduler.step()
        realLearningRate = scheduler.get_last_lr()[0]
        # wandb
        wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': valid_loss, 'precision': precision, 'f1_score': f1_score, 'sensitivity': sensitivity, 'LearningRate':realLearningRate, 'dice':dice})

        # if the model improves, save a checkpoint at this epoch
        if f1_score > best_f1:
            best_f1 = f1_score
            # 使用了多GPU需要加上module
            torch.save(model.module, model_path)
            print('saving model with best_f1 {:.5f}'.format(best_f1))

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


    