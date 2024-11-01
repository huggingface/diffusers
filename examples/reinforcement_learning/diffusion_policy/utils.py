import numpy as np


def create_sample_indices(episode_ends: np.ndarray, sequence_length: int,
                         pad_before: int = 0, pad_after: int = 0):
    """
    Creates valid indices for sampling sequences from episodes.

    Example:
    If we have episode_ends = [100, 200] (meaning episode 1 is 0-99, episode 2 is 100-199)
    And we want sequence_length = 16
    pad_before = 2 (obs_horizon-1)
    pad_after = 8 (action_horizon-1)

    For episode 1:
    - Can start 2 steps before (pad_before)
    - Can end 8 steps after (pad_after)
    - Returns indices like:
      [-2, 14, 2, 16]  # start at -2, get data 0-14, pad first 2 steps
      [-1, 15, 1, 16]  # start at -1, get data 0-15, pad first 1 step
      [0, 16, 0, 16]   # normal case
      [1, 17, 0, 16]   # normal case
      ... and so on
    """
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
    return np.array(indices)

def sample_sequence(train_data, sequence_length: int,
                   buffer_start_idx: int, buffer_end_idx: int,
                   sample_start_idx: int, sample_end_idx: int):
    """
    Gets actual data sequence using the indices from create_sample_indices.
    Handles padding for sequences at episode boundaries.

    Example:
    If indices are [-2, 14, 2, 16]:
    - Tries to get data from index -2 to 14
    - But since -2 is invalid:
      - Creates array of length 16
      - Copies first available data (index 0) to fill first 2 positions
      - Puts actual data from 0-14 in positions 2-16

    train_data = {
        'action': array of shape [total_timesteps, action_dim],
        'state': array of shape [total_timesteps, state_dim]
    }
    Returns same structure but with padded sequences
    """
    result = {}
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    """
    Computes min and max for normalization.

    Example:
    data = array of shape [1000, 2] (1000 timesteps, 2D actions)
    returns {
        'min': array([min_x, min_y]),  # minimum value for each dimension
        'max': array([max_x, max_y])   # maximum value for each dimension
    }
    """
    data = data.reshape(-1, data.shape[-1])
    return {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }

def normalize_data(data, stats):
    """Normalize data to [-1, 1] range"""
    return 2.0 * (data - stats['min']) / (stats['max'] - stats['min']) - 1.0

def unnormalize_data(data, stats):
    """Convert from [-1, 1] back to original range"""
    return (data + 1.0) / 2.0 * (stats['max'] - stats['min']) + stats['min']
