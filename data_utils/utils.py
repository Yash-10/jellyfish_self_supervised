# Author: Yash Gondhalekar

import random
import math
import torch.utils.data
from collections import defaultdict

import numpy as np
from torch.utils.data import WeightedRandomSampler


def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels


def get_imbalanced_sampler(dataset):
    dataset_indices = [d[1] for d in dataset.samples]
    # dataset_indices = dataset.indices
    y = [dataset.targets[i] for i in dataset_indices]
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])

    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

    return sampler