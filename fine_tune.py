import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
from self_supervised.model import SimCLR
from pretrain import reset_weights, train_simclr, print_options

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from data_utils.dataset_folder import prepare_data_for_pretraining

from self_supervised.constants import NUM_WORKERS, DEVICE
from self_supervised.linear_evaluation import perform_linear_eval, prepare_data_features


class FineTuneSimCLR(pl.LightningModule):
    def __init__(self, simclr_network, mlp_classifier, lr_simclr, lr_mlp, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.simclr_network = simclr_network
        self.mlp_classifier = mlp_classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        images, labels = batch

        op = simclr_network(images)
        outputs = mlp_classifier(op)
        loss = criterion(outputs, labels)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer1 = optim.AdamW(self.simclr_network.parameters(),
                                lr=lr_simclr,
                                weight_decay=self.hparams.weight_decay)
        optimizer2 = optim.AdamW(self.mlp_classifier.parameters(),
                                lr=lr_mlp,
                                weight_decay=self.hparams.weight_decay)

        return [optimizer1, optimizer2], [lr_scheduler1, lr_scheduler2]  # todo: add lr_schduler for both.


def fine_tune(batch_size, epochs=100, **kwargs):
  pl.seed_everything(42) # To be reproducible
  model = SimCLR(max_epochs=epochs, **kwargs)
  trainer.fit(model, train_loader, val_loader)
  model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets fine-tuning hyperparameters')
    parser.add_argument('--fine_tune_logistic_lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--fine_tune_logistic_batch_size', type=float, default=16, help='logistic batch size')
    parser.add_argument('--train_dir_path', type=str, default=None, help='Path to fine-tuning training directory (must end as "train/")')
    parser.add_argument('--test_dir_path', type=str, default=None, help='Path to fine-tuning training directory (must end as "test/")')
    parser.add_argument('--fine_tune_batch_size', type=int, default=128, help='batch size to use')
    parser.add_argument('--fine_tune_hidden_dim', type=int, default=128, help='hidden dimension for the MLP projection head')
    parser.add_argument('--fine_tune_lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--fine_tune_temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--fine_tune_weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--fine_tune_epochs', type=int, default=300, help='no. of fine-tuning epochs')
    parser.add_argument('--model_path', type=str, default='simclr_pretrained_model_cv.pth', help='path of the saved pretrained model to be used during fine-tuning')
    parser.add_argument('--fine_tuned_simclr_model_save_path', type=str, default='simclr_finetuned.pth', help='path to save the fine-tuned simclr model.')
    parser.add_argument('--fine_tuned_logistic_model_save_path', type=str, default='logistic_finetuned.pth', help='path to save the fine-tuned logistic model.')
    parser.add_argument('--wandb_projectname', type=str, default='crossval-my-wandb-project', help='project name for wandb logging')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    # Create a wandb logger
    wandb_logger = WandbLogger(name=f'(finetune)-simclr-{opt.fine_tune_lr}-{opt.fine_tune_logistic_lr}-{opt.fine_tune_batch_size}-{opt.fine_tune_logistic_batch_size}-{opt.fine_tune_weight_decay}-{opt.fine_tune_epochs}', project=opt.wandb_projectname)

    simclr_model = SimCLR.load_from_checkpoint(checkpoint_path=opt.model_path)

    # Fine-tuning on all non-jellyfish + JClass 3 and 4, i.e. using all data except JClass 1 and 2. Before running the script, ensure no JClass 1 or 2 images are present in the train directory.
    train_data, test_data = prepare_data_for_pretraining(opt.train_dir_path, opt.test_dir_path, mode='pretraining')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=opt.fine_tune_batch_size, pin_memory=True, num_workers=NUM_WORKERS
    )

    simclr_model.train()
    simclr_model.convnet.fc = nn.Identity()
    mlp_classifier = nn.Linear(512, 2, bias=True)

    finetune_simclr = FineTuneSimCLR(
        simclr_model.convnet,
        mlp_classifier,
        lr_simclr=1e-4,
        lr_mlp=1e-3,
        weight_decay=1e-3
    )

    ## Perform fine-tuning
    simclr_model.train()
    simclr_model.convnet.fc = nn.Identity()
    simclr_model = simclr_model.to(DEVICE)

    mlp_classifier = nn.Linear(512, 1, bias=True)
    mlp_classifier = mlp_classifier.to(DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss()

    # Since we do not use weight decay here, AdamW and Adam are actually the same, hence we use Adam instead of AdamW.
    simclr_optimizer = optim.Adam(simclr_model.convnet.parameters(), lr=opt.fine_tune_lr)  # Not using weight decay in fine-tuning - since the SimCLR paper mentions not using any regularization in fine-tuning.
    mlp_optimizer = optim.Adam(mlp_classifier.parameters(), lr=opt.fine_tune_logistic_lr)  # Not using weight decay in fine-tuning

    # imbalanced_sampler = get_imbalanced_sampler(train_feats_simclr)

    simclr_scheduler = MultiStepLR(simclr_optimizer, milestones=[10, 20], gamma=0.1)
    mlp_scheduler = MultiStepLR(mlp_optimizer, milestones=[10, 20], gamma=0.1)

    for epoch in range(opt.fine_tune_epochs):
        losses = []
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            simclr_optimizer.zero_grad()
            mlp_optimizer.zero_grad()

            op = simclr_model.convnet(images)
            outputs = mlp_classifier(op).squeeze()

            # _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels.float())

            loss.backward()
            simclr_optimizer.step()
            mlp_optimizer.step()

            simclr_scheduler.step()
            mlp_scheduler.step()

            losses.append(loss.item())
        print(f'Loss at epoch {epoch+1}: {np.mean(losses)}')

    torch.save(simclr_model.state_dict(), opt.fine_tuned_simclr_model_save_path)
    torch.save(mlp_classifier.state_dict(), opt.fine_tuned_logistic_model_save_path)
