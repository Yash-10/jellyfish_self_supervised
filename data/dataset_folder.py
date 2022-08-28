##### Reference: https://stackoverflow.com/a/50260776 #####
import numpy as np
from astropy.io import fits
from torchvision.datasets import DatasetFolder
import torch.nn.functional as F  # Note: This is different from `torchvision.transforms.functional`.

class NpyFolder(DatasetFolder):

    EXTENSIONS = ['.npy']

    def __init__(self, root, transform=None, target_transform=None,
                 loader=None, normalize_mode=None):
        if loader is None:
            loader = self.__fits_loader

        super(NpyFolder, self).__init__(root, loader, self.EXTENSIONS[0],  # 0th index corresponds to FITS.
                                         transform=transform,
                                         target_transform=target_transform)

    @staticmethod
    def __fits_loader(filename):
        data = np.load(filename).astype(np.float32)  # convert to float32 to prevent endian errors while converting to pytorch tensor.
        data = torch.from_numpy(data)
        _shape, dim1, dim2 = data.shape[0], data.shape[1], data.shape[2]
        if normalize_mode == 'across_channels':
            data = F.normalize(data, dim=0).squeeze()  # Across channels
        elif normalize_mode == "each_channel_separately":
            data = F.normalize(data.reshape(_shape, -1), dim=1).squeeze().reshape(_shape, dim1, dim2)  # Each channel separately normalize.
        else:
            pass

        return data
