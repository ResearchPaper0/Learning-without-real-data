import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

import utils
import config
from model import EBUnet
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

def get_gaussian_kernel(k=train_data.resize//16*2+1, mu=0, sigma=0.5, normalize=True):
    """Get a gaussian kernel to use an a point heatmap

    Parameters
    ----------
    k: int
        size of the border of the kernel
    mu: float
        mean of the heatmap
    sigma: float
        standard deviation of the heatmap
    normalize: bool
        to get the heatmap between 0 and 1

    Returns
    -------
    np.array
        Heatmap image of the gaussian kernel
    """
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian

    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi *sigma **2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

GAUSS_KERNEL = get_gaussian_kernel()

def get_mask(img_shape, kernel, pos):
    """ Create a mask with heatmap given a probabilistic kernel
    at a specific location associated to an image

    Parameters
    ----------
    img_shape: tuple
        shape of the image to create the corresponding mask
    kernel: np.array
        probabilistic kernel giving the heatmap
    pos: list of tuple
        position of the kernel top left position on mask

    Returns
    -------
    np.array
        mask with the kernel at pos position.
    """
    mask = np.zeros(img_shape)
    x, y = pos
    # Image ranges
    y1, y2 = max(0, y), min(img_shape[0], y + kernel.shape[0])
    x1, x2 = max(0, x), min(img_shape[1], x + kernel.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(kernel.shape[0], img_shape[0] - y)
    x1o, x2o = max(0, -x), min(kernel.shape[1], img_shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return mask

    # overlay kernel on the mask
    mask[y1:y2, x1:x2] = kernel[y1o:y2o, x1o:x2o]
    mask = mask / np.max(mask)

    return mask

def get_keypoints_mask(image, keypoints):
    """Convert the keypoints coordinates into masks of edge-based heatmap

    Parameters
    ----------
    image: torch.Tensor
        shape of the image to create the corresponding mask
    keypoints: torch.Tensor
        probabilistic kernel giving the heatmap

    Returns
    -------
    torch.Tensor
        Tensor of the heatmap mask corresponding to the keypoints
    """
    # get all the different dimensions
    B, C, H, W = image.shape
    H_g, W_g = GAUSS_KERNEL.shape
    image_dim = np.array([H, W])
    kernel_dim = np.array([H_g, W_g])

    # set the keypoints and the masks
    masks = np.zeros((B, 4, H, W))
    kpts = keypoints.detach().cpu().numpy()

    # get all the mask for the B samples and the C coordinates
    for b in range(B):
        for c in range(4):
            # get the top left position of the kernel heatmap
            # to overlay with the mask and get the mask
            x, y = (kpts[b, c] * image_dim - kernel_dim / 2 + 0.5).astype(np.int)
            masks[b, c] = get_mask((H, W), GAUSS_KERNEL, (x, y))

        # get the contours of the id cards to have an edged-heatmap
        edge = cv2.drawContours(np.zeros((H, W, 4)) , np.array([(kpts[b]* np.array([H, W])).astype(np.int)]), 0, [1]*4, 2)
        edge = edge.transpose((2, 0, 1))

        # point-wise multiplication of the two heatmapts to get
        # an edge-based heatmap on the corners
        masks[b] = masks[b] * edge

    return torch.from_numpy(masks)

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
    num_batches = int(len(dataloader))
    for i, data in tqdm(enumerate(dataloader), total=num_batches):
        counter += 1
        image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
        # get the mask from the keypoints
        true_mask = get_keypoints_mask(image, keypoints).type(torch.FloatTensor).to(config.DEVICE)

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
            image, keypoints = data['image'].to(config.DEVICE), data['keypoints'].to(config.DEVICE)
            # get the mask from keypoints
            true_mask = get_keypoints_mask(image, keypoints).type(torch.FloatTensor).to(config.DEVICE)

            pred_mask = model(image)
            loss = criterion(pred_mask, true_mask)
            valid_running_loss += loss.item()

            # retrieve the predicted keypoints from the mask
            pred_keypoints = utils.mask_to_keypoints(pred_mask)
            pred_keypoints = pred_keypoints.view(pred_keypoints.size(0), -1)

            if (epoch+1) % 100 == 0 and i < 3 + 1:
                # plot the key points masks at this epoch
                imgs=[]
                img = image.detach().cpu().numpy()[0].transpose((1, 2, 0))
                for c in range(4):
                    true = true_mask.detach().cpu().numpy()[0, c]
                    pred = pred_mask.detach().cpu().numpy()[0, c]
                    mask = np.zeros_like(img)
                    mask[..., 0] = true
                    mask[..., 1] = pred
                    imgs.append(cv2.addWeighted(img, 0.5, mask, 0.5, 0))
                imgs = np.vstack((np.hstack((imgs[0], imgs[1])), np.hstack((imgs[2], imgs[3]))))
                plt.imshow(imgs)
                plt.savefig(f"{config.OUTPUT_PATH}/kpts_mask_epoch_{epoch+1}_{i}.jpg")
                plt.close()

                # plot the predicted validation keypoints after every...
                utils.plot_val_batch(data, pred_keypoints, f'{epoch+1}_{i}', n=2)

    valid_loss = valid_running_loss / counter
    return valid_loss

if __name__ == '__main__':

    # model
    model = EBUnet().to(config.DEVICE)
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
        train_epoch_loss = fit(model, train_loader)
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
                        }, f"{config.OUTPUT_PATH}/model_ebunet_random_DA_3K.pth")

        # print current state
        print(f"Train Loss: {train_epoch_loss:.6f}")
        print(f'Val Loss: {val_epoch_loss:.6f}')
        print(f'Best Loss: {best_val_loss:.6f}')
        np.savetxt(f'{config.OUTPUT_PATH}/model_ebunet_random_DA_3K_train_loss.csv', np.array(train_loss), fmt='%f', newline=',')
        np.savetxt(f'{config.OUTPUT_PATH}/model_ebunet_random_DA_3K_val_loss.csv', np.array(val_loss), fmt='%f', newline=',')

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
