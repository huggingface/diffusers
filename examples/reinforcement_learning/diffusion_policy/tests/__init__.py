import os
import sys

import numpy as np
import torch


# Get the root directory (parent of diffusion_policy)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from diffusion_policy.config import DataConfig
from diffusion_policy.dataset import SequentialDataset


def main():
    # Get the diffusion_policy directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "pusht_cchi_v7_replay.zarr")

    print(f"Looking for dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    # Rest of your code...

if __name__ == "__main__":
    main()
