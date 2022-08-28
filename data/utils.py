import random
import math
import torch.utils.data
from collections import defaultdict

def round_up_to_even(f):
    return math.ceil(f / 2.) * 2

def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    """
    Code credit: https://gist.github.com/Alvtron/9b9c2f870df6a54fda24dbd1affdc254

    ** This version is slightly modified that uses the `round_up_to_even` function defined above. **

    """
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round_up_to_even(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels