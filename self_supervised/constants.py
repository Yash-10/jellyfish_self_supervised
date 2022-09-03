import os

# No. of workers in dataloader.
NUM_WORKERS = os.cpu_count()

# Path to the folder where the pretrained models are saved.
CHECKPOINT_PATH = "./saved_models/"  # TODO: Add it as an argument.