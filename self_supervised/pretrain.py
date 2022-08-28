import argparse

## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

from torchvision import transforms
from torch.utils import data

from model import SimCLR
import pytorch_lightning as pl

from transforms import ContrastiveTransformations
from data.dataset_folder import NpyFolder
from data.transforms import CustomColorJitter

from linear_evaluation import LogisticRegression, prepare_data_features, train_logreg


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
# DATASET_PATH = CONTRASTIVE_DATA
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
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

def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         gpus=1 if str(device)=='cuda:0' else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),  # TODO: top-1 or top-5 accuracy for binary classification
                                    LearningRateMonitor('epoch')],
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = data.DataLoader(train_data_contrast, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

        pl.seed_everything(42) # To be reproducible
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model

def print_options(opt):
    print("------------ Options ------------")
    # TODO
    print("------------ End ------------")

if __name__ == "__main":
    parser = argparse.ArgumentParser(description='sets pretraining hyperparameters')
    parser.add_argument('--batch_size', default=64, help='batch size to use')
    parser.add_argument('--lr', default=5e-4, help='learning rate')
    parser.add_argument('--temperature', default=0.07, help='temperature')
    parser.add_argument('--weight_decay', default=1e-4, help='weight decay')
    parser.add_argument('--max_epochs', default=300, help='no. of pretraining epochs')
    parser.add_argument('--model_save_path', default='simclr_pretrained_model.pth', help='path to save the pretrained model')
    parser.add_argument('--train_dataset_frac', default=1., help='fraction of dataset to use for pre-training (it performs stratified splitting to preseve class ratio)')
    parser.add_argument('--to_linear_eval', default=False, help='Whether to perform linear evaluation after pre-training')
    parser.add_argument('--to_fine_tune', default=False, help='Whether to fine-tune after pre-training')
    parser.add_argument('--logistic_lr', default=1e-3, help='Learning rate for logistic regression training on features. Only used if to_linear_eval=True')
    parser.add_argument('--logistic_weight_decay', default=1e-3, help='Weight decay for logistic regression training on features. Only used if to_linear_eval=True')
    parser.add_argument('--logistic_batch_size', default=32, help='Batch size for logistic regression training on features. Only used if to_linear_eval=True')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    # Calculate mean and std of each channel across training dataset.
    dataset = DummyNpyFolder('train', transform=None)  # train
    loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    mean, std = get_mean_std(loader)

    contrast_transforms = transforms.Compose([
                        # torchvision.transforms.RandomApply([
                        #     CustomGaussNoise(),                                  
                        # ], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.CenterCrop(size=200),
                        transforms.RandomResizedCrop(size=96),
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
                        transforms.GaussianBlur(kernel_size=9),  # This is an important augmentation -- else results were considerably worse!
                        # transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
    ])
    # transforms.RandomPerspective(p=0.5) turned out to be unhelpful since performance decreased.

    # Create data
    unlabeled_data = NpyFolder('train', transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # train
    train_data_contrast = NpyFolder('test', transform=ContrastiveTransformations(contrast_transforms, n_views=2))  # test

    # Use below code if want to stratify split both datasets. TODO: Add code for handling fraction of dataset...
    # unlabeled_data = stratified_split(unlabeled_data, labels=unlabeled_data.targets, fraction=0.2, random_state=123)
    # train_data_contrast = stratified_split(train_data_contrast, labels=train_data_contrast.targets, fraction=0.2, random_state=123)
    # unlabeled_data[0].targets = unlabeled_data[1]
    # train_data_contrast[0].targets = train_data_contrast[1]

    # unlabeled_data = unlabeled_data[0]
    # train_data_contrast = train_data_contrast[0]

    simclr_model = train_simclr(batch_size=opt.batch_size,
                            hidden_dim=opt.hidden_dim,
                            lr=opt.lr,
                            temperature=opt.temperature,
                            weight_decay=opt.weight_decay,
                            max_epochs=opt.max_epochs)

    torch.save(simclr_model.state_dict(), opt.model_save_path)

    if opt.to_linear_eval:
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