import scipy
import numpy as np
import torch

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

def get_gaussian_noise_scale_per_channel(train_dir, num_channels=12):
    mad = np.empty((num_channels,))
    for channel in range(num_channels):
        data = []
        for img_class in glob.glob(train_dir+'/*'):
            for img in glob.glob(img_class + '/*.npy'):
                flattened_img = np.load(img).reshape(num_channels, -1)[channel]
                data.append(flattened_img[flattened_img != 0.])

        mad[channel] = scipy.stats.median_abs_deviation(
            np.hstack(data)
        )
    return mad

class CustomGaussNoise(object):
    def __init__(self, mean=0.):
        self.mean = mean
        mad = get_gaussian_noise_scale_per_channel('train')
        scaling = np.random.uniform(low=1., high=3.)
        gauss_noise_stds = torch.from_numpy(scaling * mad)
        self.std = gauss_noise_stds

    def __call__(self, img):
        """
        Call custom Gaussian noise transformation
        """
        noises = torch.empty(size=(self.std.shape[0], 350, 350))
        for i in range(self.std.shape[0]):
            noises[i] = torch.normal(mean=self.mean, std=gauss_noise_stds[i], size=(350, 350))
        return img + noises

    def __repr__(self):
        return self.__class__.__name__ + '()'

class CustomRandGrayscale(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        """
        :param img: image

        :return: mean-stacked pixel-wise, image
        """
        if torch.rand(1) < self.p:
            return torch.mean(img, axis=0).unsqueeze(0)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class CustomColorJitter(object):
    """
    This transformation was described in the paper: https://doi.org/10.3390/rs13112181
    """
    def __init__(self):
        pass

    def __call__(self, img):
        """
        :param img: image

        :return: jittered image
        """
        n_channels = img.shape[0]
        return torch.mul(
            torch.FloatTensor(n_channels, 1, 1).uniform_(0.8, 1.2),
            img
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"
