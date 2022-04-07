import torch
import cv2
import pandas as pd
import numpy as np

import torch

import config
import utils

from transform import MEDIUM_TRANSFORM


def train_test_split(root, split):
    """Create a train/test split to train and evalutate a model

    Parameters
    ----------
    root: str
        path root of the dataset folder
    split: int
        split ratio for the test set

    Returns
    -------
    tuple
        Tuple of training set and testing set in pd.Dataframe format
    """
    # convert the dataset into a dataframe
    # where to find the 4*2 points coordinates
    df_data = utils.id_card_data_set_to_pandas(root)
    df_shuffle = df_data.sample(frac=1) #on prend 5% pour tester le LR 0.95,random_state=26
    len_data = len(df_data)

    # calculate the validation data sample length
    valid_split = int(len_data * split)

    # calculate the training data samples length
    train_split = int(len_data - valid_split)

    # split the samples
    training_samples = df_shuffle.iloc[:train_split][:]
    valid_samples = df_shuffle.iloc[-valid_split:][:]
    return training_samples, valid_samples

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class to process the training phase of models
    """

    def __init__(self, samples, path, transform=MEDIUM_TRANSFORM, resize=128):
        """Initialize an Dataset generator of the front of ID card

        Parameters
        ----------
        samples: pd.DataFrame
            dataframe of the sample to compose the dataset
        path: str
            root path folder where the data are stored
        transform: list
            transformations to apply before returning the samples

        Returns
        -------
        Dataset
            The Dataset given the images and the coordinates of the id card
        """
        self.path = path
        self.data = samples
        self.transform = transform

        # 128 for EB-UNet and 448 for KPR
        self.resize = resize

    def __len__(self):
        """Get the length of the dataset

        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """Get a sample of (image, target) for training process

        Parameters
        ----------
        index: int
            index of the sample to return

        Returns
        -------
        tuple
            transormed image and target (flatten keypoints) of the ID card corners
        """

        try:
            temp_path = f"{self.path}/{self.data.index[index]}"
            #temp_path = temp_path.split('.')[0] + ".jpg"
            image = cv2.imread(temp_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) * 1.
        except:
            print("temp_path :", temp_path)
            print("path : ", self.path)
            print("index : ", index)
            print("data : ", self.data)
            print("data index : ", self.data.index[index])
        # get the keypoints
        keypoints = self.data.iloc[index]
        keypoints = np.array(keypoints, dtype='float32')
        # reshape the keypoints
        keypoints = keypoints.reshape(-1, 2)

        for t in self.transform:
            image, keypoints = t(image, keypoints)

        orig_h, orig_w, channel = image.shape
        # resize the image into `resize` defined above
        image = cv2.resize(image, (self.resize, self.resize))
        # again reshape to add grayscale channel format
        image = image / 255.0
        # transpose for getting the channel size to index 0
        image = np.transpose(image, (2, 0, 1))

        keypoints = keypoints / np.array([orig_w, orig_h])

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'keypoints': torch.tensor(keypoints, dtype=torch.float)
        }

# get the training and validation data samples
training_samples, valid_samples = train_test_split(
    config.ROOT_PATH,
    config.TEST_SPLIT
)

# initialize the dataset
train_data = Dataset(training_samples, f"{config.ROOT_PATH}/X/front")
valid_data = Dataset(valid_samples, f"{config.ROOT_PATH}/X/front")

# prepare data loaders
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=config.BATCH_SIZE,
    shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_data,
    batch_size=config.BATCH_SIZE,
    shuffle=False
)

print(f"Training sample instances: {len(train_data)}")
print(f"Validation sample instances: {len(valid_data)}")

# whether to show dataset keypoint plots
if config.SHOW_DATASET_PLOT:
    utils.plot_dataset_batch(valid_data, n=4)
