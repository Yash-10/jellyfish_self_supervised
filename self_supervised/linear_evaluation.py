import os
from copy import deepcopy
from turtle import window_height
from tqdm.notebook import tqdm

import pytorch_lightning as pl

# Setting the seed
pl.seed_everything(42)

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F

from data_utils.dataset_folder import NpyFolder
from data_utils.calculate_mean_std import DummyNpyFolder, get_mean_std

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from self_supervised.constants import CHECKPOINT_PATH, DEVICE
from self_supervised.model import SimCLR
from self_supervised.evaluation import print_classification_report, plot_confusion_matrix
from data_utils.utils import get_imbalanced_sampler

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


NUM_WORKERS = os.cpu_count()

# class LogisticRegression(pl.LightningModule):

#     def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100, weight=torch.tensor([0.3, 0.7]).to(DEVICE)):
#         super().__init__()
#         self.save_hyperparameters()
#         # Mapping from representation h to classes
#         self.model = nn.Linear(feature_dim, num_classes)
#         self.weight = weight

#     def configure_optimizers(self):
#         # optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
#         optimizer = optim.AdamW(self.parameters(),
#                                 lr=self.hparams.lr,
#                                 weight_decay=self.hparams.weight_decay)
#         lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
#                                                       milestones=[int(self.hparams.max_epochs*0.1),
#                                                                   int(self.hparams.max_epochs*0.8)],
#                                                       gamma=0.1)
#         return [optimizer], [lr_scheduler]

#     def _calculate_loss(self, batch, mode='train'):
#         feats, labels = batch
#         preds = self.model(feats)
#         loss = F.cross_entropy(preds, labels, weight=self.weight)
#         acc = (preds.argmax(dim=-1) == labels).float().mean()

#         self.log(mode + '_loss', loss)
#         self.log(mode + '_acc', acc)
#         return loss

#     def training_step(self, batch, batch_idx):
#         return self._calculate_loss(batch, mode='train')

#     def validation_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='val')

#     def test_step(self, batch, batch_idx):
#         self._calculate_loss(batch, mode='test')


@torch.no_grad()
def prepare_data_features(model, dataset, device=DEVICE):
    """Given a pre-trained model, it returns the features extracted from a dataset."""
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    if isinstance(dataset, data.Dataset):
        data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    else:  # Assume already a dataloader
        data_loader = dataset

    feats, labels, batch_images = [], [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs.float())
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)
        batch_images.append(batch_imgs)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    batch_images = torch.cat(batch_images, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels), batch_images


class Logistic_Reg_model(torch.nn.Module):
    def __init__(self, no_input_features):
        super(Logistic_Reg_model, self).__init__()
        self.layer1 = torch.nn.Linear(no_input_features, 32)
        self.layer2 = torch.nn.Linear(32, 1)
    def forward(self, x):
        y_predicted = self.layer1(x)
        y_predicted = self.layer2(y_predicted)
        return y_predicted

def perform_linear_eval(
    train_feats_simclr, test_feats_simclr, number_of_epochs=100, lr=1e-2, weight_decay=1e-3
):
    """_summary_

    Args:
        train_img_loader (_type_): _description_
        test_img_loader (_type_): _description_
        simclr_model (_type_): _description_

    Returns:
        _type_: _description_

    Note: train_img_loader and test_img_loader can either be dataloaders or can also be a dataset object.

    """
    test_feats, test_labels = (
        test_feats_simclr.tensors[0].numpy(), test_feats_simclr.tensors[1].numpy()
    )

    imbalanced_sampler = get_imbalanced_sampler(train_feats_simclr)
    train_loader = torch.utils.data.DataLoader(train_feats_simclr, batch_size=16, sampler=imbalanced_sampler)
    # test_loader = torch.utils.data.DataLoader(test_feats_simclr, batch_size=16)

    model = Logistic_Reg_model(512)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)
    for epoch in range(number_of_epochs):
        for x, y in train_loader:
            y_prediction = model(x)
            loss = criterion(y_prediction.float().squeeze(), y.float().squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if (epoch + 1) % 10 == 0:
            print('epoch:', epoch + 1, ', loss = ', loss.item())

    with torch.no_grad():
        y_pred = torch.sigmoid(model(test_feats))
        y_pred_class = y_pred.round()

    print_classification_report(test_labels, y_pred_class)
    plot_confusion_matrix(test_labels, y_pred_class)

    return y_pred_class, y_pred, test_labels
