import torch
import zarr
from config import DataConfig

from utils import create_sample_indices, get_data_stats, normalize_data, sample_sequence


class SequentialDataset(torch.utils.data.Dataset):
    """Dataset for loading and processing sequential data"""

    def __init__(self, config: DataConfig):
        self.config = config

        # load data from zarr dataset
        dataset_root = zarr.open(config.dataset_path, 'r')

        # load training data
        self.train_data = {
            'action': dataset_root['data']['action'][:],
            'state': dataset_root['data']['state'][:],
        }
        if hasattr(dataset_root['data'], 'image'):
            self.train_data['image'] = dataset_root['data']['image'][:]

        # get episode endings
        self.episode_ends = dataset_root['meta']['episode_ends'][:]

        # create sample indices
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=config.pred_horizon,
            pad_before=config.obs_horizon-1,
            pad_after=config.action_horizon-1
        )

        # compute statistics and normalize data
        self.stats = {}
        self.normalized_data = {}
        for key, data in self.train_data.items():
            self.stats[key] = get_data_stats(data)
            self.normalized_data[key] = normalize_data(data, self.stats[key])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get sequence indices
        buffer_start_idx, buffer_end_idx, \
        sample_start_idx, sample_end_idx = self.indices[idx]

        # sample normalized sequences
        sample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.config.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # prepare observations (both state and/or image)
        result = {'action': sample['action']}

        if 'state' in sample:
            result['state'] = sample['state'][:self.config.obs_horizon]

        if 'image' in sample:
            result['image'] = sample['image'][:self.config.obs_horizon]

        return result
