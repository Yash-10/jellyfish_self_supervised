##### Reference: https://stackoverflow.com/a/50260776 #####
import numpy as np
from astropy.io import fits
import torch
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F  # Note: This is different from `torchvision.transforms.functional`.

class NpyFolder(DatasetFolder):

    EXTENSIONS = ['.npy']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, normalize_mode=None):
        if loader is None:
            loader = self.__fits_loader

        super(NpyFolder, self).__init__(root, loader, self.EXTENSIONS[0],  # 0th index corresponds to .npy
                                         transform=transform,
                                         target_transform=target_transform)
        self.normalize_mode = normalize_mode

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
