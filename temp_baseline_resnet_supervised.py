from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from data_utils.calculate_mean_std import DummyNpyFolder, get_mean_std
from data_utils.transformations import CustomColorJitter
from data_utils.dataset_folder import NpyFolder

from self_supervised.evaluation import *

class ResNet(pl.LightningModule):

    def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet34(
            pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.7),
                                                                  int(self.hparams.max_epochs*0.9)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        preds_labels = torch.argmax(preds, axis=-1)
        prec, recall, f1, _ = precisionRecallFscoreSupport(labels.numpy(), preds_labels.numpy())
        self.log(mode + '_prec', prec)
        self.log(mode + '_recall', recall)
        self.log(mode + '_f1_score', f1)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


print('Calculating mean and standard deviation across training dataset...')
dataset = DummyNpyFolder('/content/train', transform=None)  # train
loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
mean, std = get_mean_std(loader)

img_transforms = transforms.Compose([
                        # torchvision.transforms.RandomApply([
                        #     CustomGaussNoise(),                                  
                        # ], p=0.5),
                        transforms.CenterCrop(size=200),
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

train_img_data = NpyFolder('/content/train', transform=img_transforms)  # train
test_img_data = NpyFolder('/content/test', transform=img_transforms)  # test

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from self_supervised.constants import CHECKPOINT_PATH, DEVICE
from self_supervised.constants import NUM_WORKERS
import os

def train_resnet(batch_size, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
                         gpus=1 if str(DEVICE)=="cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1,
                         check_val_every_n_epoch=2)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_img_data, batch_size=batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = data.DataLoader(test_img_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    pl.seed_everything(42) # To be reproducable
    model = ResNet(**kwargs)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint('resnet.ckpt')
    model = ResNet.load_from_checkpoint('resnet.ckpt')

    # Test best model on validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0], "test": val_result[0]}

    return model, result

resnet_model, resnet_result = train_resnet(batch_size=8,
                                           num_classes=2,
                                           lr=1e-3,
                                           weight_decay=2e-4,
                                           max_epochs=1)
print(resnet_result)
