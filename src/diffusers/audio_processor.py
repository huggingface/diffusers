# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union

import numpy as np
import torch


PipelineAudioInput = Union[
    np.ndarray,
    torch.Tensor,
    List[np.ndarray],
    List[torch.Tensor],
]


def is_valid_audio(audio) -> bool:
    r"""
    Checks if the input is a valid audio.

    A valid audio can be:
    - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

    Args:
        audio (`Union[np.ndarray, torch.Tensor]`):
            The audio to validate. It can be a NumPy array or a torch tensor.

    Returns:
        `bool`:
            `True` if the input is a valid audio, `False` otherwise.
    """
    return isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim in (2, 3)


def is_valid_audio_audiolist(audios):
    r"""
    Checks if the input is a valid audio or list of audios.

    The input can be one of the following formats:
    - A 4D tensor or numpy array (batch of audios).
    - A valid single audio: 2D `np.ndarray` or `torch.Tensor` (grayscale audio), 3D `np.ndarray` or `torch.Tensor`.
    - A list of valid audios.

    Args:
        audios (`Union[np.ndarray, torch.Tensor, List]`):
            The audio(s) to check. Can be a batch of audios (4D tensor/array), a single audio, or a list of valid
            audios.

    Returns:
        `bool`:
            `True` if the input is valid, `False` otherwise.
    """
    if isinstance(audios, (np.ndarray, torch.Tensor)) and audios.ndim == 4:
        return True
    elif is_valid_audio(audios):
        return True
    elif isinstance(audios, list):
        return all(is_valid_audio(audio) for audio in audios)
    return False
