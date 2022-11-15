import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import config
from model import Unet
from dataset import train_loader, valid_loader, train_data


def dice_bce_loss(target, pred, smooth=1e-1):
    """Custom loss to train the EB-UNet.
    Sum of the Dice loss and the Binary Cross-Entropy loss with mean reduction

    Parameters
    ----------
    target: torch.Tensor
        target of the input
    pred: torch.Tensor
        predicition of the model for the input
    smooth: float
        smoothing value for the dice loss

    Returns
    -------
    torch.Tensor
        Dice + BCE losses mean-reduced
    """
    batch_size = target.shape[0]
    # flatten the target and pred
    target = target.view(batch_size, -1)
    pred = pred.view(batch_size, -1)

    # get the values for the dice loss
    interserction = (target * pred).sum(1)
    total = (target + pred).sum(1)

    # compute and return the loss
    dice = (2 * interserction + smooth) / (total + smooth)
    return (1 - dice.mean()) + nn.BCELoss()(target, pred)


def fit(model, dataloader, epoch):
    """Fit model on one epoch of a dataloader

    Parameters
    ----------
    model: torch.nn.Module
        model to train
    dataloader: torch.util.data.DataLoader
        DataLoader

    Returns
    -------
    float
        train loss on the epoch
    """
    print('Training')
    model.train()
    train_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(dataloader))
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, true_mask = data['image'].to(config.DEVICE), data['mask'].to(config.DEVICE)
        # pass forward
        optimizer.zero_grad()
        pred_mask = model(image)

        # pass backward
        loss = criterion(pred_mask, true_mask)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

    train_loss = train_running_loss/counter
    return train_loss

def validate(model, dataloader, epoch):
    """Validate model on one epoch of a dataloader

    Parameters
    ----------
    model: torch.nn.Module
        model to validate
    dataloader: torch.util.data.DataLoader
        DataLoader
    epoch: object
        to convert to str to construct the file name of the save plot

    Returns
    -------
    float
        validation loss on the epoch
    """
    model.eval()
    valid_running_loss = 0.0
    counter = 0
    # calculate the number of batches
    num_batches = int(len(dataloader))
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=num_batches):
            counter += 1
            image, true_mask = data['image'].to(config.DEVICE), data['mask'].to(config.DEVICE)

            pred_mask = model(image)
            loss = criterion(pred_mask, true_mask)
            valid_running_loss += loss.item()

    valid_loss = valid_running_loss / counter
    return valid_loss

if __name__ == '__main__':

    # model
    model = Unet().to(config.DEVICE)
    if config.FROM_CHECKPOINT:
        print('Loading checkpoint...')
        # load the model checkpoint
        checkpoint = torch.load(f'{config.OUTPUT_PATH}/model.pth')
        # load model weights state_dict
        model.load_state_dict(checkpoint['model_state_dict'])

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # LR was 1e-2
    criterion = dice_bce_loss


    # training loop
    train_loss = []
    val_loss = []

    best_val_loss = 1e6
    best_model = None
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1} of {config.EPOCHS}")
        # train part
        train_epoch_loss = fit(model, train_loader, epoch)
        train_loss.append(train_epoch_loss)

        # validation aprt
        val_epoch_loss = validate(model, valid_loader, epoch)
        val_loss.append(val_epoch_loss)

        # save the model if the validation is the best one
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                        'epoch': config.EPOCHS,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, f"{config.OUTPUT_PATH}/model_unet_ALDA_3K.pth")

        # print current state
        print(f"Train Loss: {train_epoch_loss:.6f}")
        print(f'Val Loss: {val_epoch_loss:.6f}')
        print(f'Best Loss: {best_val_loss:.6f}')
        np.savetxt(f'{config.OUTPUT_PATH}/model_unet_ALDA_3K.csv', np.array(train_loss), fmt='%f', newline=',')
        np.savetxt(f'{config.OUTPUT_PATH}/model_unet_ALDA_3K.csv', np.array(val_loss), fmt='%f', newline=',')

        # Early stopping when no improvement
        if len(val_loss) > config.EARLY_STOP_N and min(val_loss[-config.EARLY_STOP_N:]) > best_val_loss:
            break



    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print('DONE TRAINING')
