# Author: Yash Gondhalekar

##### Reference: https://stackoverflow.com/a/50260776 #####
import numpy as np
from astropy.io import fits

import torch
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F  # Note: This is different from `torchvision.transforms.functional`.

from data_utils.transformations import ContrastiveTransformations, CustomColorJitter
from data_utils.calculate_mean_std import DummyNpyFolder, get_mean_std


class NpyFolder(DatasetFolder):

    EXTENSIONS = ['.npy']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None):
        if loader is None:
            loader = self.__fits_loader

        super(NpyFolder, self).__init__(root, loader, self.EXTENSIONS[0],  # 0th index corresponds to .npy
                                         transform=transform,
                                         target_transform=target_transform)

    @staticmethod
    def __fits_loader(filename):
        data = np.load(filename).astype(np.float32)  # convert to float32 to prevent endian errors while converting to pytorch tensor.
        data = torch.from_numpy(data)
        # ** The below commented code can be used to either normalize images across channels or each channel separately.
        # _shape, dim1, dim2 = data.shape[0], data.shape[1], data.shape[2]
        # if self.normalize_mode == 'across_channels':
        #     data = F.normalize(data, dim=0).squeeze()  # Across channels
        # elif self.normalize_mode == "each_channel_separately":
        #     data = F.normalize(data.reshape(_shape, -1), dim=1).squeeze().reshape(_shape, dim1, dim2)  # Each channel separately normalize.
        # elif self.normalize_mode is None:  # This is the default. In this case, normalization is done from the augmentation pipeline.
        #     pass

        return data

def prepare_data_for_pretraining(train_dir_path, test_dir_path=None, mode='pretraining'):
    """_summary_

    Args:
        train_dir_path (_type_): _description_
        test_dir_path (_type_): _description_
        mode (str, optional): Either 'pretraining' or 'cv'. Defaults to 'pretraining'.

    Returns:
        _type_: _description_
    """
    if test_dir_path is None and mode == 'pretraining':  # Pretraining needs a test directory but cv does not.
        raise ValueError("No test directory supplied for pretraining!")

    # Calculate mean and std of each channel across training dataset.
    print('Calculating mean and standard deviation across training dataset...')
    dataset = DummyNpyFolder(train_dir_path, transform=None)  # train
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    contrast_transforms = transforms.Compose([
                        # torchvision.transforms.RandomApply([
                        #     CustomGaussNoise(),                                  
                        # ], p=0.5),
                        transforms.CenterCrop(size=200),
                        transforms.RandomResizedCrop(size=72),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply([
                            CustomColorJitter()
                        ], p=0.8
                        ),
                        # CustomRandGrayscale(p=0.2),
                        # transforms.RandomPerspective(p=0.3),
                        transforms.RandomRotation(degrees=(0, 360)),
                        # transforms.RandomApply([
                        #     transforms.ColorJitter(brightness=0.5,
                        #                            contrast=0.5,
                        #                            saturation=0.5,
                        #                            hue=0.1)
                        # ], p=0.8),
                        transforms.RandomApply([
                            transforms.GaussianBlur(kernel_size=9)  # This is an important augmentation -- else results were considerably worse!
                        ], p=0.5
                        ),
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
    ])
    # transforms.RandomPerspective(p=0.5) turned out to be unhelpful since performance decreased.

    # Create data
    train_data = NpyFolder(train_dir_path, transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # train
    if mode == 'pretraining':
        test_data = NpyFolder(test_dir_path, transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # test
    else:
        test_data = None

    return train_data, test_data
