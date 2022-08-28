import pytorch_lightning as pl

# Setting the seed
pl.seed_everything(42)

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data.dataset_folder import NpyFolder


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
def prepare_data_features(model, dataset):
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

def train_logreg(batch_size, train_feats_data, test_feats_data, max_epochs=100, **kwargs):
    """Trains a logistic regression model on the extracted features."""
    # TODO: Do K-Fold cross validation instead.
    model_suffix = "jellyfish"  # Any suffix would do.
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         gpus=1 if str(device)=="cuda:0" else 0,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),  # TODO: Use a diff monitor.
                                    LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate=0,
                         check_val_every_n_epoch=10)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=1, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result

def test_logreg(logreg_model, test_feats_simclr):
    """Test the fitted logistic regression model on test features.
    
    logreg_model: LogisticRegression
        A trained logisitc regression model.
    test_feats_simclr: torch.tensor
        Features of test images.
    
    Returns
    -------
    preds_labels: numpy.ndarray
        Predicted labels
    preds: numpy.ndarray
        Predicted probabilities
    true_labels: numpy.ndarray
        True labels

    """
    shuffled_dataset = torch.utils.data.Subset(test_feats_simclr, torch.randperm(len(test_feats_simclr)).tolist())
    test_loader = data.DataLoader(shuffled_dataset, batch_size=1, num_workers=0, shuffle=False)
    
    outputs, true_labels = [], []
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
