import argparse
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data_utils.calculate_mean_std import DummyNpyFolder, get_mean_std
from data_utils.transformations import CustomColorJitter
from data_utils.dataset_folder import NpyFolder

from self_supervised.evaluation import precisionRecallFscoreSupport
from pretrain import print_options

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from self_supervised.constants import CHECKPOINT_PATH, DEVICE
from self_supervised.constants import NUM_WORKERS
from cross_validate import _stratify_works_as_expected
from sklearn.model_selection import StratifiedKFold
import os


class ResNet(pl.LightningModule):

    def __init__(self, num_classes, lr, weight_decay, max_epochs=100, weight=torch.tensor([0.5, 0.5]).to(DEVICE)):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.weight = weight

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
        loss = F.cross_entropy(preds, labels, weight=self.weight)
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


def train_resnet(batch_size, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
                         gpus=1 if str(DEVICE)=="cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_img_data, batch_size=batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    pl.seed_everything(42) # To be reproducable
    model = ResNet(**kwargs)
    trainer.fit(model, train_loader)
    trainer.save_checkpoint('resnet_supervised.ckpt')
    model = ResNet.load_from_checkpoint('resnet_supervised.ckpt')

    # Get performance on train set.
    train_result = trainer.test(model, train_loader, verbose=False)
    result = {"train": train_result[0]}

    return model, trainer, result


def kfold_stratified_cross_validate_simclr(
    training_dataset, batch_size, lr, weight_decay,
    k_folds=3, num_epochs=100, model_save_path="Resnet_supervised.ckpt", logger=None
):
    # Set fixed random number seed
    torch.manual_seed(42)

    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # ** Perform cross validation **
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(training_dataset, training_dataset.targets)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = data.SubsetRandomSampler(train_ids)
        test_subsampler = data.SubsetRandomSampler(test_ids)

        # Ensure there is no data leakage.
        assert set(train_subsampler.indices).isdisjoint(set(test_subsampler.indices)), "Data leakage while performing K-Fold cross validation!"

        print(f'No. of training examples: {len(train_subsampler.indices)}, No. of testing examples: {len(test_subsampler.indices)}')

        # Define data loaders for training and testing data in this fold.
        train_loader = data.DataLoader(
                        training_dataset, batch_size=batch_size, sampler=train_subsampler,
                        pin_memory=True, num_workers=NUM_WORKERS)
        test_loader = data.DataLoader(
                        training_dataset,
                        batch_size=batch_size, sampler=test_subsampler)

        # Ensure stratified split works as expected.
        print("Train loader class distribution:")
        _stratify_works_as_expected(data.DataLoader(training_dataset))
        print("[fold] train loader class distribution:")
        _stratify_works_as_expected(train_loader)
        print("[fold] test loader class distribution:")
        _stratify_works_as_expected(test_loader)

        resnet_model, trainer, _ = train_resnet(batch_size=batch_size,
                                            num_classes=2,
                                            lr=lr,
                                            weight_decay=weight_decay,
                                            max_epochs=num_epochs)

        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        test_loader = data.DataLoader(test_img_data, batch_size=opt.batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        cross_val_test_result = trainer.test(resnet_model, test_loader, verbose=False)
        loss, acc, prec, recall, f1_score = cross_val_test_result[0]["crossval_test_loss"], cross_val_test_result[0]["crossval_test_acc"], cross_val_test_result[0]["crossval_test_prec"], cross_val_test_result[0]["crossval_test_recall"], cross_val_test_result[0]["crossval_test_f1_score"]

        # Print result on this fold.
        print(f'Result on fold {fold}: {[loss, acc, prec, recall, f1_score]}')
        print('--------------------------------')
        results[fold] = [loss, acc, prec, recall, f1_score]

        # clear
        del resnet_model, trainer

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_loss, sum_acc, sum_prec, sum_recall, sum_f1_score = 0.0, 0.0, 0.0, 0.0, 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum_loss += value[0]
        sum_acc += value[1]
        sum_prec += value[2]
        sum_recall += value[3]
        sum_f1_score += value[4]

    assert len(results.items()) == k_folds
    avg_loss = sum_loss / k_folds
    avg_acc = sum_acc / k_folds
    avg_prec = sum_prec / k_folds
    avg_recall = sum_recall / k_folds
    avg_f1_score = sum_f1_score / k_folds
    print(f'Average test metrics (loss, acc, prec, recall, f1_score) over all folds: {avg_loss, avg_acc, avg_prec, avg_recall, avg_f1_score}')

    return avg_loss, avg_acc, avg_prec, avg_recall, avg_f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets pretraining hyperparameters for cross-validation')
    parser.add_argument('--train_dir_path', type=str, default=None, help='Path to training directory (must end as "train/")')
    parser.add_argument('--k_folds', type=int, default=3, help='number of folds to create for cross validation.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--max_epochs', type=int, default=300, help='no. of pretraining epochs')
    parser.add_argument('--model_save_path', type=str, default='resnet_supervised.pth', help='path to save the trained model during cross-validation')
    parser.add_argument('--wandb_projectname', type=str, default='crossval-my-wandb-project', help='project name for wandb logging')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    # Create a wandb logger
    wandb_logger = WandbLogger(name=f'supervised_resnet-{opt.batch_size}-{opt.lr}-{opt.weight_decay}-{opt.max_epochs}', project=opt.wandb_projectname)  # For each distinct set of hyperparameters, use a different `name`.

    print('Calculating mean and standard deviation across training dataset...')
    dataset = DummyNpyFolder(opt.train_dir_path, transform=None)  # train
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    img_transforms = transforms.Compose([
                            transforms.CenterCrop(size=200),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomApply([
                                CustomColorJitter()
                            ], p=0.8
                            ),
                            transforms.RandomRotation(degrees=(0, 360)),
                            transforms.RandomApply([
                                transforms.GaussianBlur(kernel_size=9)  # This is an important augmentation -- else results were considerably worse!
                            ], p=0.5
                            ),
                            transforms.Normalize(mean=mean, std=std)
        ])

    train_img_data = NpyFolder(opt.train_dir_path, transform=img_transforms)  # train

    avg_loss, avg_acc, avg_prec, avg_recall, avg_f1_score = kfold_stratified_cross_validate_simclr(
        train_img_data, opt.batch_size, opt.lr, opt.weight_decay,
        k_folds=3, num_epochs=100, model_save_path="Resnet_supervised.ckpt", logger=None
    )
    wandb.log({"avg_loss": avg_loss})
    wandb.log({"avg_acc": avg_acc})
    wandb.log({"avg_prec": avg_prec})
    wandb.log({"avg_recall": avg_recall})
    wandb.log({"avg_f1_score": avg_f1_score})
