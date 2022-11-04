# For argument parsing.
import argparse

## Standard libraries
import os
from copy import deepcopy

## tqdm for loading bars
from tqdm.notebook import tqdm

from torch.utils import data

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from self_supervised.model import SimCLR
from data_utils.transformations import ContrastiveTransformations
from data_utils.dataset_folder import NpyFolder
from data_utils.transformations import CustomColorJitter
from self_supervised.constants import NUM_WORKERS, CHECKPOINT_PATH, DEVICE
from data_utils.dataset_folder import prepare_data_for_pretraining

from self_supervised.linear_evaluation import perform_linear_eval, prepare_data_features
from self_supervised.evaluation import print_classification_report, plot_confusion_matrix


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from data_utils.calculate_mean_std import get_mean_std, DummyNpyFolder


def reset_weights(m):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m: module.

    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_simclr(train_loader, batch_size, max_epochs=500, save_path=None, logger=None, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         gpus=1 if str(DEVICE)=='cuda:0' else 0,
                         max_epochs=max_epochs,
                         callbacks=[LearningRateMonitor('epoch')],  # TODO: top-1 or top-5 accuracy for binary classification
                         progress_bar_refresh_rate=1,
                         logger=logger)

    pl.seed_everything(42) # To be reproducible
    model = SimCLR(max_epochs=max_epochs, **kwargs)
    model.apply(reset_weights)
    trainer.fit(model, train_loader)  # No validation loader is present during training since KFold cross validation is used.

    # Save model.
    if save_path is None:
        save_path = "SimCLR.ckpt"
    trainer.save_checkpoint(save_path)
    model = SimCLR.load_from_checkpoint(checkpoint_path=save_path)

    return model, trainer


def print_options(opt):
    print('\n')
    print("------------ Options ------------")
    for arg in vars(opt):
        print(f'{arg}:\t\t{getattr(opt, arg)}')
    print("------------ End ------------")
    print('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets pretraining hyperparameters')
    parser.add_argument('--train_dir_path', type=str, default=None, help='Path to training directory (must end as "train/")')
    parser.add_argument('--test_dir_path', type=str, default=None, help='Path to testing directory (must end as "test/")')
    parser.add_argument('--encoder', type=str, default='resnet34', help='encoder architecture to use. Options: resnet18 | resnet34 | resnet52')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for the MLP projection head')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--max_epochs', type=int, default=300, help='no. of pretraining epochs')
    parser.add_argument('--model_save_path', type=str, default='simclr_pretrained_model.pth', help='path to save the pretrained model')
    parser.add_argument('--train_dataset_frac', type=float, default=1., help='fraction of dataset to use for pre-training (it performs stratified splitting to preseve class ratio)')
    parser.add_argument('--wandb_projectname', type=str, default='my-wandb-project', help='project name for wandb logging')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    if opt.train_dir_path is None or opt.test_dir_path is None:
        raise ValueError("No train directory supplied!")

    # Create a wandb logger
    wandb_logger = WandbLogger(name=f'pretrain-simclr-{opt.lr}-{opt.temperature}-{opt.weight_decay}-{opt.max_epochs}', project=opt.wandb_projectname)

    train_data, test_data = prepare_data_for_pretraining(opt.train_dir_path, opt.test_dir_path, mode='pretraining')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.batch_size, pin_memory=True, num_workers=NUM_WORKERS
    )

    # Perform pretraining.
    simclr_model, trainer = train_simclr(
        train_loader,
        batch_size=opt.batch_size,
        hidden_dim=opt.hidden_dim,
        lr=opt.lr,
        temperature=opt.temperature,
        weight_decay=opt.weight_decay,
        max_epochs=opt.max_epochs,
        save_path=opt.model_save_path,
        logger=wandb_logger,
        encoder=opt.encoder
    )

    # We re-create the datasets since while extracting the features, we do not need contrastive transforms, but only normalization.
    # Calculate mean and std of each channel across training dataset.
    print('Pretraining finished. Starting to extract features...')
    print('Calculating mean and standard deviation across training dataset...')
    dataset = DummyNpyFolder(opt.train_dir_path, transform=None)
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    img_transforms = transforms.Compose([
        transforms.Normalize(mean, std)
    ])

    train_img_data = NpyFolder(opt.train_dir_path, transform=img_transforms)  # train
    test_img_data = NpyFolder(opt.test_dir_path, transform=img_transforms)  # test

    # Extract features
    train_feats_simclr, train_batch_images = prepare_data_features(simclr_model, train_img_data)
    test_feats_simclr, test_batch_images = prepare_data_features(simclr_model, test_img_data)

    # Save everything.
    torch.save(train_feats_simclr, 'train_feats_simclr.pt')
    torch.save(test_feats_simclr, 'test_feats_simclr.pt')
    torch.save(train_batch_images, 'train_batch_images.pt')
    torch.save(test_batch_images, 'test_batch_images.pt')
