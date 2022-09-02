# For argument parsing.
import argparse

## Standard libraries
import os
from copy import deepcopy

## tqdm for loading bars
from tqdm.notebook import tqdm

from torchvision import transforms
from torch.utils import data

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold

from self_supervised.model import SimCLR
from data.transformations import ContrastiveTransformations
from data.dataset_folder import NpyFolder
from data.transformations import CustomColorJitter

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# DATASET_PATH = CONTRASTIVE_DATA
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"  # TODO: Add it as an argument.
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

from data.calculate_mean_std import get_mean_std, DummyNpyFolder
from data.utils import stratified_split


def reset_weights(m):
    """Try resetting model weights to avoid weight leakage.

    Args:
        m: module.

    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def _stratify_works_as_expected(loader):
    zero_label = 0
    one_label = 0
    for _, y in loader:
        for l in y:
            if l == 0:
                zero_label += 1
            elif l == 1:
                one_label += 1

    print(f'No. of examples with label 0: {zero_label}, with label 1: {one_label}')

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

def test_simclr(trainer, test_loader, model_path):
    cross_val_test_result = trainer.test(dataloaders=test_loader, ckpt_path=model_path)
    return cross_val_test_result[0]["crossval_test_loss"]

def kfold_stratified_cross_validate(
    training_dataset, batch_size, hidden_dim, lr, temperature, weight_decay,
    k_folds=5, num_epochs=300, model_save_path="SimCLR_pretrained.ckpt"
):
    # Set fixed random number seed
    torch.manual_seed(42)

    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # ** Perform cross validation **
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(training_dataset, training_dataset.targets)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Ensure there is no data leakage.
        assert set(train_subsampler.indices).isdisjoint(set(test_subsampler.indices)), "Data leakage while performing K-Fold cross validation!"

        print(f'No. of training examples: {len(train_subsampler.indices)}, No. of testing examples: {len(test_subsampler.indices)}')

        # Define data loaders for training and testing data in this fold.
        train_loader = torch.utils.data.DataLoader(
                        training_dataset, batch_size=batch_size, sampler=train_subsampler,
                        pin_memory=True, num_workers=NUM_WORKERS)
        test_loader = torch.utils.data.DataLoader(
                        training_dataset,
                        batch_size=batch_size, sampler=test_subsampler)

        # Ensure stratified split works as expected.
        print("Train loader class distribution:")
        _stratify_works_as_expected(torch.utils.data.DataLoader(training_dataset))
        print("[fold] train loader class distribution:")
        _stratify_works_as_expected(train_loader)
        print("[fold] test loader class distribution:")
        _stratify_works_as_expected(test_loader)

        save_path = os.path.join(CHECKPOINT_PATH, model_save_path)
        simclr_model, trainer = train_simclr(train_loader,
                            batch_size=batch_size,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            temperature=temperature,
                            weight_decay=weight_decay,
                            max_epochs=num_epochs,
                            save_path=save_path
                        )

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        result = test_simclr(trainer, test_loader, model_path=save_path)

        # Print result on this fold.
        print(f'Result on fold {fold}: {result}')
        print('--------------------------------')
        results[fold] = result

        # clear
        del simclr_model, trainer

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum += value

    avg_result = sum / len(results.items())
    print(f'Average test metric over all folds: {avg_result}')

    return avg_result

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

    # Calculate mean and std of each channel across training dataset.
    print('Calculating mean and standard deviation across training dataset...')
    dataset = DummyNpyFolder(opt.train_dir_path, transform=None)  # train
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    contrast_transforms = transforms.Compose([
                        # torchvision.transforms.RandomApply([
                        #     CustomGaussNoise(),                                  
                        # ], p=0.5),
                        transforms.CenterCrop(size=200),
                        transforms.RandomResizedCrop(size=96),
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
    train_data = NpyFolder(opt.train_dir_path, transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # train
    test_data = NpyFolder(opt.test_dir_path, transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # test

    # K-Fold cross validation.
    print('Starting K-Fold cross validation...')
    avg_loss = kfold_stratified_cross_validate(
        train_data, opt.batch_size, opt.hidden_dim, opt.lr, opt.temperature, opt.weight_decay, k_folds=5, num_epochs=opt.max_epochs,
        model_save_path=opt.model_save_path
    )

    print(f'\n\nAverage metric value with current hyperparameters: {avg_loss}\n\n')

    # Run pretraining with best hyperparameters found using cross validation.

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