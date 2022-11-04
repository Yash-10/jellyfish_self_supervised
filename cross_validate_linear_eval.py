
import argparse
import wandb

import torch
from pytorch_lightning.loggers import WandbLogger
from pretrain import print_options

from sklearn.model_selection import StratifiedKFold

from torch.utils import data
from self_supervised.linear_evaluation import perform_linear_eval
from self_supervised.constants import CHECKPOINT_PATH, NUM_WORKERS
from self_supervised.evaluation import precisionRecallFscoreSupport


def kfold_cv(train_feats_simclr, k_folds=3, lr=1e-2, num_epochs=100, batch_size=1):
    # Set fixed random number seed
    torch.manual_seed(42)

    # For fold results
    results = {}

    # Define the K-fold Cross Validator
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    # ** Perform cross validation **
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_feats_simclr, train_feats_simclr.tensors[1])):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = data.SubsetRandomSampler(train_ids)
        test_subsampler = data.SubsetRandomSampler(test_ids)

        # Ensure there is no data leakage.
        assert set(train_subsampler.indices).isdisjoint(set(test_subsampler.indices)), "Data leakage while performing K-Fold cross validation!"

        print(f'No. of training examples: {len(train_subsampler.indices)}, No. of testing examples: {len(test_subsampler.indices)}')

        xx = torch.utils.data.TensorDataset(train_feats_simclr.tensors[0][train_ids], train_feats_simclr.tensors[1][train_ids])
        yy = torch.utils.data.TensorDataset(train_feats_simclr.tensors[0][test_ids], train_feats_simclr.tensors[1][test_ids])
        y_pred_class, _, test_labels = perform_linear_eval(
            xx, yy, number_of_epochs=num_epochs, lr=lr
        )

        precision, recall, f1_score, _ = precisionRecallFscoreSupport(test_labels, y_pred_class)

        # Print result on this fold.
        print(f'Result on fold {fold}: {precision, recall, f1_score}')
        print('--------------------------------')
        results[fold] = [precision, recall, f1_score]


    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_prec, sum_recall, sum_f1_score = 0.0, 0.0, 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum_prec += value[0]
        sum_recall += value[1]
        sum_f1_score += value[2]

    assert len(results.items()) == k_folds
    avg_prec = sum_prec / k_folds
    avg_recall = sum_recall / k_folds
    avg_f1_score = sum_f1_score / k_folds
    print(f'Average test metrics (prec, recall, f1-score) over all folds: {avg_prec, avg_recall, avg_f1_score}')

    return avg_prec, avg_recall, avg_f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets linear evaluation hyperparameters for cross-validation')
    parser.add_argument('--train_feats_path', type=str, default=None, help='Path to training representations')
    parser.add_argument('--k_folds', type=int, default=3, help='no. of folds.')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='no. of epochs')
    parser.add_argument('--wandb_projectname', type=str, default='crossval-my-wandb-project', help='project name for wandb logging')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    # Create a wandb logger
    wandb_logger = WandbLogger(name=f'{opt.k_folds}-{opt.batch_size}-{opt.lr}-{opt.num_epochs}', project=opt.wandb_projectname)  # For each distinct set of hyperparameters, use a different `name`.

    train_feats_simclr = torch.load(opt.train_feats_path)
    avg_prec, avg_recall, avg_f1_score = kfold_cv(
        train_feats_simclr, k_folds=opt.k_folds, lr=opt.lr,
        num_epochs=opt.num_epochs, batch_size=opt.batch_size
    )
    wandb.log({"avg_prec": avg_prec})
    wandb.log({"avg_recall": avg_recall})
    wandb.log({"avg_f1_score": avg_f1_score})
