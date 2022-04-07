import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import config
from model import KPR
from dataset import train_loader, valid_loader


def fit(model, dataloader):
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
    num_batches = int(len(dataloader)/dataloader.batch_size)
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # flatten the keypoints
        keypoints = keypoints.view(keypoints.size(0), -1)

        # pass forward
        optimizer.zero_grad()
        outputs = model(image)

     

        # pass backward
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        train_running_loss += loss.item()

           
    train_loss = train_running_loss/counter
    return train_loss

def validate(model, dataloader, epoch=''):
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
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # flatten the keypoints
            keypoints = keypoints.view(keypoints.size(0), -1)

            outputs = model(image)
            loss = criterion(outputs, keypoints)
            valid_running_loss += loss.item()

            # plot the predicted validation keypoints...
            if (epoch+1) % 100 == 0 and i == 0:
                utils.plot_val_batch(data, outputs, f'{epoch+1}_0', n=1)
                utils.plot_val_batch(data, outputs, f'{epoch+1}_1', n=1)
                utils.plot_val_batch(data, outputs, f'{epoch+1}_2', n=1)
                utils.plot_val_batch(data, outputs, f'{epoch+1}_3', n=1)
                

    valid_loss = valid_running_loss/counter
    return valid_loss


if __name__ == '__main__':
    # model
    model = KPR(pretrained=True, requires_grad=False).to(config.DEVICE)
    if config.FROM_CHECKPOINT:
        print('Loading checkpoint...')
        # load the model checkpoint
        checkpoint = torch.load(f'{config.OUTPUT_PATH}/model.pth')
        # load model weights state_dict
        model.load_state_dict(checkpoint['model_state_dict'])


    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    # LR was 1e-3
    # we need a loss function which is good for regression like SmmothL1Loss ...
    # ... or MSELoss
    criterion = nn.MSELoss()

    # training loop
    train_loss = []
    val_loss = []

    best_val_loss = 1e6
    best_model = None

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1} of {config.EPOCHS}")
        # train part
        train_epoch_loss = fit(model, train_loader)
        train_loss.append(train_epoch_loss)

        # validation part
        val_epoch_loss = validate(model, valid_loader, epoch) 
        val_loss.append(val_epoch_loss)

        # save the model if the validation is the best one
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({
                        'epoch': config.EPOCHS,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                        }, f"{config.OUTPUT_PATH}/model_kpr_DA_3K.pth")

        # print current state
        print(f"Train Loss: {train_epoch_loss:.6f}")
        print(f'Val Loss: {val_epoch_loss:.6f}')
        print(f'Best Loss: {best_val_loss:.6f}')
        np.savetxt(f'{config.OUTPUT_PATH}/model_kpr_DA_3K_train_loss.csv', np.array(train_loss), fmt='%f', newline=',')
        np.savetxt(f'{config.OUTPUT_PATH}/model_kpr_DA_3K_val_loss.csv', np.array(val_loss), fmt='%f', newline=',')

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
