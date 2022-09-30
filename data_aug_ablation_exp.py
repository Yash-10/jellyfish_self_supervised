import wandb
import os
import argparse
import torch
from sklearn.model_selection import StratifiedKFold

from torch.utils import data
from pytorch_lightning.loggers import WandbLogger

from pretrain import train_simclr, print_options
from self_supervised.constants import CHECKPOINT_PATH, NUM_WORKERS
from self_supervised.linear_evaluation import perform_linear_eval
from self_supervised.evaluation import (
    print_classification_report, plot_confusion_matrix, precisionRecallFscoreSupport
)
from data_utils.dataset_folder import prepare_data_for_pretraining


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


def test_simclr(trainer, test_loader, model_path):
    cross_val_test_result = trainer.test(dataloaders=test_loader, ckpt_path=model_path)
    return cross_val_test_result[0]["crossval_test_loss"], cross_val_test_result[0]["crossval_test_acc_top5"]


def kfold_stratified_cross_validate_simclr(
    train_dir_path, training_dataset, batch_size, hidden_dim, lr, temperature, weight_decay,
    logistic_lr, logistic_weight_decay, logistic_batch_size,
    k_folds=3, num_epochs=100, model_save_path="SimCLR_pretrained_dataaug_exp.ckpt", logger=None, encoder='resnet32'
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

        save_path = os.path.join(CHECKPOINT_PATH, model_save_path)
        simclr_model, trainer = train_simclr(train_loader,
                            batch_size=batch_size,
                            hidden_dim=hidden_dim,
                            lr=lr,
                            temperature=temperature,
                            weight_decay=weight_decay,
                            max_epochs=num_epochs,
                            save_path=save_path,
                            logger=logger,
                            encoder=encoder
                        )
        simclr_model.eval()  # Set it to eval mode.

        _, preds_labels, _, true_labels = perform_linear_eval(
            train_dir_path, hidden_dim, lr, temperature, weight_decay, num_epochs,
            logistic_lr, logistic_weight_decay, logistic_batch_size, simclr_model
        )
        print(f'Final linear evaluation results:')
        print('Classification report:')
        print_classification_report(true_labels, preds_labels)
        print('Confusion matrix')
        plot_confusion_matrix(true_labels, preds_labels)
        precision, recall, f1_score, support = precisionRecallFscoreSupport(true_labels, preds_labels)

        # Print result on this fold.
        print(f'Result on fold {fold} (precision, recall, f1_score, support): {precision, recall, f1_score, support}')
        print('--------------------------------')
        results[fold] = [precision, recall, f1_score, support]

        # clear
        del simclr_model, trainer

    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1_score = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value}')
        sum_precision += value[0]
        sum_recall += value[1]
        sum_f1_score += value[2]

    assert len(results.items()) == k_folds
    avg_precision = sum_precision / k_folds
    avg_recall = sum_recall / k_folds
    avg_f1_score = sum_f1_score / k_folds
    print(f'Average test metrics (precision, recall, f1_score) over all folds: {avg_precision, avg_recall, avg_f1_score}')

    return avg_precision, avg_recall, avg_f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets pretraining hyperparameters for cross-validation (data augmentation ablation experiment)')
    parser.add_argument('--train_dir_path', type=str, default=None, help='Path to training directory (must end as "train/")')
    parser.add_argument('--k_folds', type=int, default=3, help='number of folds to create for cross validation.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimension for the MLP projection head')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--temperature', type=float, default=0.07, help='temperature')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--logistic_lr', type=float, default=1e-4, help='learning rate for logistic regression')
    parser.add_argument('--logistic_weight_decay', type=float, default=1e-4, help='weight decay for logistic regression')
    parser.add_argument('--logistic_batch_size', type=float, default=1e-4, help='batch size for logistic regression')
    parser.add_argument('--max_epochs', type=int, default=300, help='no. of pretraining epochs')
    parser.add_argument('--encoder', type=str, default='resnet18', help='encoder architecture to use. Options: resnet18 | resnet34 | resnet52')
    parser.add_argument('--model_save_path', type=str, default='simclr_pretrained_model_cv.pth', help='path to save the pretrained model during cross-validation')
    parser.add_argument('--wandb_projectname', type=str, default='crossval-my-wandb-project', help='project name for wandb logging')

    opt = parser.parse_args()

    # Print options
    print_options(opt)

    # Create a wandb logger
    wandb_logger = WandbLogger(name=f'{opt.encoder}-{opt.batch_size}-{opt.hidden_dim}-{opt.lr}-{opt.temperature}-{opt.weight_decay}-{opt.max_epochs}', project=opt.wandb_projectname)  # For each distinct set of hyperparameters, use a different `name`.

    # Prepare data.
    train_data, _ = prepare_data_for_pretraining(opt.train_dir_path, mode='cv')

    # K-Fold cross validation.
    print('Starting K-Fold cross validation...')
    avg_precision, avg_recall, avg_f1_score = kfold_stratified_cross_validate_simclr(
        opt.train_dir_path, train_data, opt.batch_size, opt.hidden_dim, opt.lr, opt.temperature, opt.weight_decay,
        opt.logistic_lr, opt.logistic_weight_decay, opt.logistic_batch_size,
        k_folds=opt.k_folds, num_epochs=opt.max_epochs, model_save_path=opt.model_save_path,
        logger=wandb_logger, encoder=opt.encoder
    )

    print(f'\n\nAverage metric value with current hyperparameters (avg_precision, avg_recall, avg_f1_score): {avg_precision, avg_recall, avg_f1_score}\n\n')
    wandb.log({"avg_precision": avg_precision})
    wandb.log({"avg_recall": avg_recall})
    wandb.log({"avg_f1_score": avg_f1_score})
