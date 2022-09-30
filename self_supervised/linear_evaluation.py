import os
from copy import deepcopy
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


NUM_WORKERS = os.cpu_count()

class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        # optimizer = optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.1),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')


@torch.no_grad()
def prepare_data_features(model, dataset, device=DEVICE):
    """Given a pre-trained model, it returns the features extracted from a dataset."""
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
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


def train_logreg(batch_size, train_feats_data, max_epochs=100, save_path='LogisticRegression_jellyfish.ckpt', **kwargs):
    """Trains a logistic regression model on the extracted features."""
    model_suffix = "jellyfish"  # Any suffix would do.
    ckpt_path = os.path.join(CHECKPOINT_PATH, save_path)
    trainer = pl.Trainer(default_root_dir=ckpt_path,
                         gpus=1 if str(DEVICE)=="cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=0)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader)
        model = LogisticRegression.load_from_checkpoint(checkpoint_path=ckpt_path)

    # Test best model on train set.
    train_result = trainer.test(model, train_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"]}

    return model, result


def perform_linear_eval(
    train_dir_path, logistic_lr, logistic_weight_decay, logistic_batch_size, simclr_model, num_epochs=100
):
    # Calculate mean and std of each channel across training dataset.
    print('Calculating mean and standard deviation across training dataset...')
    dataset = DummyNpyFolder(train_dir_path, transform=None)  # train
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    # For linear evaluation, no transforms are used apart from normalization.
    img_transforms = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    # Note: This is the same dataset as pretraining, but with no transforms.
    # TODO: Set seed so that same loader is used as in pretraining -- IMP.
    train_img_data = NpyFolder(train_dir_path, transform=img_transforms)  # train

    # Extract features
    train_feats_simclr, _ = prepare_data_features(simclr_model, train_img_data)
    feature_dim = train_feats_simclr.tensors[0].shape[1]

    logreg_model, results = train_logreg(
        batch_size=logistic_batch_size,
        train_feats_data=train_feats_simclr,
        feature_dim=feature_dim,
        num_classes=2,  # jellyfish and non-jellyfish
        lr=logistic_lr,
        weight_decay=logistic_weight_decay,  # Perform grid-search on this to find which weight decay is the best
        max_epochs=num_epochs
    )
    print(f'Linear evaluation training results:\n\t{results}')
    print('Now starting testing...')

    # Testing.
    test_img_data = NpyFolder('test', transform=img_transforms)  # test
    test_feats_simclr, test_batch_images = prepare_data_features(simclr_model, test_img_data)

    preds_labels, preds, true_labels = test_logreg(logreg_model, test_batch_images, test_feats_simclr)

    return logreg_model, preds_labels, preds, true_labels


def test_logreg(logreg_model, test_batch_images, test_feats_simclr):
    """Test the fitted logistic regression model on test features.
    
    logreg_model: LogisticRegression
        A trained logisitc regression model.
    test_feats_simclr: torch.tensor
        Features of test images.
    
    Returns
    -------
    preds_labels: numpy.ndarray
        Predicted labels of shape (num_examples,).
    preds: numpy.ndarray
        Predicted probabilities of shape (num_examples, 2).
    true_labels: numpy.ndarray
        True labels of shape (num_examples,).

    """
    shuffled_dataset = torch.utils.data.Subset(test_feats_simclr, torch.randperm(len(test_feats_simclr)).tolist())
    test_loader = data.DataLoader(shuffled_dataset, batch_size=1, num_workers=0, shuffle=False)
    
    outputs, true_labels, failed = [], [], []
    for i, (x, y) in enumerate(test_loader):
        logreg_output = logreg_model.model(x).detach().data.cpu()

        # Note: `pred` is the probability array.
        # Note: We need to add a sigmoid layer since the LogisticRegression class only outputs Ax+b. Now, pred values must be all between 0 and 1, inclusive.
        # In the LogisticRegression class we did not have a sigmoid layer at the end because while training, we used the F.cross_entropy function that internally does LogSoftmax + NLLLoss. If we were to get the probabilities manually, as done here, we need a sigmoid layer.
        # Tangential points: If we instead used nn.BCELoss instead of nn.CrossEntropyLoss inside the LogisticRegression class, then we would need the sigmoid layer at the end. To solve this for binary problems, we can use BCEWithLogitsLoss and then we don't need sigmoid at the end.
        pred = torch.sigmoid(logreg_output).numpy()

        assert pred.all() >= 0 and pred.all() <= 1

        if np.argmax(pred, axis=-1)[0] != y.item():
            failed.append((test_batch_images[i], y, np.argmax(pred, axis=-1)))

        outputs.append(pred)
        true_labels.append(y)

    preds = np.vstack(outputs)
    preds_labels = np.argmax(preds, axis=-1)
    true_labels = [t.item() for t in true_labels]

    return preds_labels, preds, true_labels
