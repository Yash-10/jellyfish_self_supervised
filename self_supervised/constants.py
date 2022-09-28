import os
import torch

# No. of workers in dataloader.
NUM_WORKERS = os.cpu_count()

# Path to the folder where the pretrained models are saved.
CHECKPOINT_PATH = "./saved_models/"

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", DEVICE)
print("Number of workers:", NUM_WORKERS)
