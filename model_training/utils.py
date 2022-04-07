import os
import json
import glob
import tqdm

import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

def draw_keypoints(image, keypoints):
    """Dran arrows in an image from one point to another
    of the ID card's corners

    Parameters
    ----------
    image: np.array
        image of the ID card
    keypoints: np.array
        keypoints of the ID card in the image

    Returns
    -------
    np.array
        Drawn image with arrows from one point to another
        of the ID card's corners in the image
    """
    image = image.copy()
    for j in range(len(keypoints)):
        # add the arrows of the true coordinates
        x1, y1 = keypoints[j]
        if j < 3:
            x2, y2 = keypoints[j+1]
            image = cv2.arrowedLine(image, (x1, y1), (x2, y2), (0.53, 0.00, 1.00), 4, tipLength = 0.05)
    return image


def plot_val_batch(data, outputs, epoch, n=2):
    """Save a plot of a validation batch

    Parameters
    ----------
    data: dict
        dictionnary of the sample batch
    outputs: torch.Tensor
        predicted flatten keypoints normalized coordinates
        of the id card in the images of the data batch
    epoch: object
        object of the epoch to convert to str in the file name of the saved plot
    n: int
        number of row/columns of sample in the batch to plot
    """
    plt.figure(figsize=(10, 10))

    # reshape the outputs as 4*2 coordinates points
    outputs = outputs.reshape(-1, 4, 2).detach().cpu().numpy()

    for i in range(n**2):
        # get the image, true and predicted points
        output = outputs[i]
        img = data['image'][i]
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0))

        h, w, _ = img.shape
        keypoints = data['keypoints'][i]
        keypoints = (keypoints * np.array([w, h])).int()
        output = (output * np.array([w, h])).astype(np.int)


        # plot the image
        plt.subplot(n, n, i+1)
        plt.imshow(img)
        for j in range(len(keypoints)):
            # add the arrows of the true coordinates
            x, y = keypoints[j]
            if j < 3:
                dx, dy = keypoints[j+1]
                dx, dy = (dx - x), (dy - y)
                plt.arrow(
                    x, y, dx, dy,
                    color='#80f',
                    head_width=5, length_includes_head=True
                )
            plt.plot(x, y, '.', color='#80f')

            # add the arrows of the predicted coordinates
            x, y = output[j]
            if j < 3:
                dx, dy = output[j+1]
                dx, dy = (dx - x), (dy - y)
                plt.arrow(
                    x, y, dx, dy,
                    color='#2f9599',
                    head_width=5, length_includes_head=True
                )

            plt.plot(x, y, '.', color='#2f9599')

        # set the plot better and save it
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f"{config.OUTPUT_PATH}/val_epoch_{epoch}.png")
    plt.close()


def plot_dataset_batch(data, n=3, transpose=True):
    """Save a plot of a validation batch

    Parameters
    ----------
    data: dict
        dictionnary of the sample batch
    n: int
        number of row/columns of sample in the batch to plot
    transpose: bool
        to transpose the C*H*W image into H*W*C image

    """
    plt.figure(figsize=(10, 10))
    for i in range(n*n):
        # get the image, true and predicted points
        sample = data[i]
      
        img = sample['image']
        img = np.array(img, dtype='float32')
        img = np.transpose(img, (1, 2, 0)) if transpose else img

        h, w, _ = img.shape
        keypoints = sample['keypoints']
        keypoints = (keypoints * np.array([w, h]))

        # plot the image
        plt.subplot(n, n, i+1)
        plt.imshow(img)
        for j in range(len(keypoints)):
            # add the arrows of the true coordinates
            x, y = keypoints[j]
            if j < 3:
                dx, dy = keypoints[j+1]
                dx, dy = (dx - x), (dy - y)
                plt.arrow(
                    x, y, dx, dy,
                    color=(1/4 + j/4, 0, 1),
                    head_width=5, length_includes_head=True
                )
            plt.plot(x, y, '.', color=(1/4 + j/4, 0, 1))
    # show the plot
    plt.show()
    plt.close()


def id_card_data_set_to_pandas(root):
    """Conver the dataset folder into a pandas DataFrame
    containing the image filename of each sample and its 8 flatten
    coordinates points of the id card borders in the image

    Parameters
    ----------
    root: str
        path root to the dataset folder

    Returns
    -------
    pd.DataFrame
        The Dataset given the images and the coordinates of the id card
    """
    # get all the dict points files in the folder
    dict_points_files = glob.glob(f'{root}/Y/dict_points/*')


    # extract the stem of the file
    dict_points_files = sorted(
        dict_points_files,
        #key=lambda x: int(x.split('\\')[-1].split('/')[-1].split('.')[0])
        key=lambda x: int(x.split('\\')[-1].split(os.sep)[-1].split('.')[0])
    )
    index = [] # stem
    columns = range(8) # number of the point
    keypoints = [] # coordinates of the points x1, y1, x2, y2, ..., y4


    for dict_points_file in tqdm.tqdm(dict_points_files):
        # set the image file name
        image_filename = dict_points_file.split(os.sep)[-1].split('.')[0] + '.png'
        index.append(image_filename)

        # Get the points
        with open(dict_points_file, 'r') as f:
            dict_points = json.loads(f.read())
        points = dict_points['corners_front_side'][0]
        points = np.array(points).reshape(-1)
        # inverted by:    keypoints = np.array(keypoints).reshape(4, 2)
        keypoints.append(points)

    # create and return the DataFrame
    index = np.array(index)
    columns = np.array(columns)
    keypoints = np.array(keypoints)
    data = pd.DataFrame(keypoints, index=index, columns=columns)
    return data


def mask_to_keypoints(masks):
    """Convert the B*C*H*W masks into B*C*2 keypoints
    extracting the max coordinates of each (B, C) masks

    Parameters
    ----------
    masks: torch.Tensor
        B*C*H*W masks of keypoints heatmap

    Returns
    -------
    torch.Tensor
        The B*C*2 keypoints of the id card according to the masks
    """
    B, C, H, W = masks.shape
    kpts = torch.zeros((B, C, 2))
    for b in range(B):
        for c in range(C):
            # get the (B, C)-mask
            bc_mask = masks[b, c].detach().cpu().numpy()
            # extract the x and y coordinates of the max
            y, x = np.unravel_index(bc_mask.argmax(), bc_mask.shape)
            # unnormalize and set the coordinate
            x, y = x/H, y/W
            kpts[b, c, 0] = x
            kpts[b, c, 1] = y
    return kpts
