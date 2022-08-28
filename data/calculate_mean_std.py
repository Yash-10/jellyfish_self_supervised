"""
This module calculates the mean and standard deviation of each channel across all images in the training dataset.
For this, it creates a dummy dataset folder *without any transforms*.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm

# Create the dataset
import numpy as np
import cv2
from astropy.io import fits
from torchvision.datasets import DatasetFolder
from PIL import Image
import torch.nn.functional as F
import skimage

class DummyNpyFolder(DatasetFolder):  # Code adapted from https://stackoverflow.com/a/50260776

    EXTENSIONS = ['.npy']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__fits_loader

        super(NpyFolder, self).__init__(root, loader, self.EXTENSIONS[0],  # 0th index corresponds to FITS.
                                         transform=transform,
                                         target_transform=target_transform)

    @staticmethod
    def __fits_loader(filename):
        data = np.load(filename).astype(np.float32)
        data = torch.from_numpy(data)
        return data

def get_mean_std(loader):
    """
    loader: torch.utils.data.DataLoader
        A Dataloader object.

    Calculates the mean and standard deviation of each channel across the training dataset.

    Notes
    -----
    Code taken from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

if __name__ == "__main__":
    dataset = DummyNpyFolder('train', transform=None)  # train
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    mean, std = get_mean_std(train_loader)
    print(f'mean: {mean}, std: {std}')