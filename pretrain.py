# For argument parsing.
import argparse

## Standard libraries
import os
from copy import deepcopy

## tqdm for loading bars
from tqdm.notebook import tqdm

from torch.utils import data

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from self_supervised.model import SimCLR
from data.transformations import ContrastiveTransformations
from data.dataset_folder import NpyFolder
from data.transformations import CustomColorJitter
from constants import NUM_WORKERS, CHECKPOINT_PATH
from cross_validate import kfold_stratified_cross_validate_simclr
from data.dataset_folder import prepare_data_for_pretraining


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

from data.calculate_mean_std import get_mean_std, DummyNpyFolder


def reset_weights(m):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m: module.

    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_simclr(train_loader, batch_size, max_epochs=500, save_path=None, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         gpus=1 if str(device)=='cuda:0' else 0,
                         max_epochs=max_epochs,
                         callbacks=[LearningRateMonitor('epoch')],  # TODO: top-1 or top-5 accuracy for binary classification
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

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
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for the MLP projection head')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--max_epochs', type=int, default=300, help='no. of pretraining epochs')
    parser.add_argument('--model_save_path', type=str, default='simclr_pretrained_model.pth', help='path to save the pretrained model')
    parser.add_argument('--train_dataset_frac', type=float, default=1., help='fraction of dataset to use for pre-training (it performs stratified splitting to preseve class ratio)')
    parser.add_argument('--to_linear_eval', type=bool, default=False, help='Whether to perform linear evaluation after pre-training')
    parser.add_argument('--to_fine_tune', type=bool, default=False, help='Whether to fine-tune after pre-training')
    parser.add_argument('--logistic_lr', type=float, default=1e-3, help='Learning rate for logistic regression training on features. Only used if to_linear_eval=True')
    parser.add_argument('--logistic_weight_decay', type=float, default=1e-3, help='Weight decay for logistic regression training on features. Only used if to_linear_eval=True')
    parser.add_argument('--logistic_batch_size', type=int, default=32, help='Batch size for logistic regression training on features. Only used if to_linear_eval=True')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    if opt.train_dir_path is None or opt.test_dir_path is None:
        raise ValueError("No train directory supplied!")

    train_data, test_data = prepare_data_for_pretraining(opt.train_dir_path, opt.test_dir_path)
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
        max_epochs=opt.num_epochs,
        save_path=opt.model_save_path
)

    if opt.to_linear_eval:
        # TODO: All below this to edit...
        from self_supervised.linear_evaluation import LogisticRegression, prepare_data_features, train_logreg

        # For linear evaluation, no transforms are used apart from normalization.
        img_transforms = transforms.Compose([
            transforms.Normalize(mean, std)
        ])

        # Load pre-trained model to use as a fixed feature extractor.
        simclr_model = SimCLR(
            hidden_dim=opt.hidden_dim, lr=opt.lr, temperature=opt.temperature, weight_decay=opt.weight_decay, max_epochs=opt.max_epochs
        )
        simclr_model.load_state_dict(torch.load(opt.model_save_path))
        simclr_model.eval()  # For pretraining, set it to eval mode.

        # Note: This is the same dataset as pretraining, but with no transforms.
        # TODO: Set seed so that same loader is used as in pretraining -- IMP.
        train_img_data = NpyFolder('train', transform=img_transforms)  # train
        test_img_data = NpyFolder('test', transform=img_transforms)  # test

        # Extract features
        train_feats_simclr, _ = prepare_data_features(simclr_model, train_img_data)
        test_feats_simclr, test_batch_images = prepare_data_features(simclr_model, test_img_data)
        feature_dim = train_feats_simclr.tensors[0].shape[1]

        logreg_model, results = train_logreg(
            batch_size=opt.logistic_batch_size,
            train_feats_data=train_feats_simclr,
            test_feats_data=test_feats_simclr,
            feature_dim=feature_dim,
            num_classes=2,  # jellyfish and non-jellyfish
            lr=opt.logistic_lr,
            weight_decay=opt.logistic_weight_decay  # Perform grid-search on this to find which weight decay is the best
        )