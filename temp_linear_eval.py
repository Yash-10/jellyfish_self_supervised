from self_supervised.linear_evaluation import perform_linear_eval, prepare_data_features
from self_supervised.evaluation import print_classification_report, plot_confusion_matrix
from torch.utils import data
from data_utils.dataset_folder import NpyFolder
from data_utils.calculate_mean_std import DummyNpyFolder, get_mean_std

from torchvision import transforms
import torch
import torch.nn as nn
from self_supervised.model import SimCLR
import pytorch_lightning as pl
import torchvision

logistic_weight_decay=1e-4
logistic_lr=1e-4
logistic_batch_size=4

print('Calculating mean and standard deviation across training dataset...')
dataset = DummyNpyFolder('/content/train', transform=None)  # train
loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=True)
mean, std = get_mean_std(loader)

img_transforms = transforms.Compose([
    transforms.Normalize(mean, std)
])

simclr_model = SimCLR(
    hidden_dim=128, lr=1e-4, temperature=0.05,
    weight_decay=1e-4, max_epochs=1000, encoder='resnet34'
)
simclr_model.load_from_checkpoint('/content/drive/MyDrive/simclr_pretrained_1000epochs.pth')
simclr_model.eval()  # Set it to eval mode.

train_img_data = NpyFolder('/content/train', transform=img_transforms)  # train
test_img_data = NpyFolder('/content/test', transform=img_transforms)  # test

train_img_loader = data.DataLoader(train_img_data, batch_size=logistic_batch_size, pin_memory=True, num_workers=2)
test_img_loader = data.DataLoader(test_img_data, batch_size=logistic_batch_size, pin_memory=True, num_workers=2)

logreg_model, preds_labels, preds, true_labels=perform_linear_eval(
    train_img_loader, test_img_loader, logistic_lr, logistic_weight_decay,
    logistic_batch_size, simclr_model, num_epochs=1000
)
print_classification_report(true_labels, preds_labels)
plot_confusion_matrix(true_labels, preds_labels)
