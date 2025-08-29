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
import PIL.Image
import torch


PipelineAudioInput = Union[
    PIL.Image.Image,  # librosa?
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],  # librosa?
    List[np.ndarray],
    List[torch.Tensor],
]


def is_valid_image(image) -> bool:
    r"""
    Checks if the input is a valid image.

    A valid image can be:
    - A `PIL.Image.Image`.
    - A 2D or 3D `np.ndarray` or `torch.Tensor` (grayscale or color image).

    Args:
        image (`Union[PIL.Image.Image, np.ndarray, torch.Tensor]`):
            The image to validate. It can be a PIL image, a NumPy array, or a torch tensor.

    Returns:
        `bool`:
            `True` if the input is a valid image, `False` otherwise.
    """
    return isinstance(image, PIL.Image.Image) or isinstance(image, (np.ndarray, torch.Tensor)) and image.ndim in (2, 3)


def is_valid_image_imagelist(images):
    r"""
    Checks if the input is a valid image or list of images.

    The input can be one of the following formats:
    - A 4D tensor or numpy array (batch of images).
    - A valid single image: `PIL.Image.Image`, 2D `np.ndarray` or `torch.Tensor` (grayscale image), 3D `np.ndarray` or
      `torch.Tensor`.
    - A list of valid images.

    Args:
        images (`Union[np.ndarray, torch.Tensor, PIL.Image.Image, List]`):
            The image(s) to check. Can be a batch of images (4D tensor/array), a single image, or a list of valid
            images.

    Returns:
        `bool`:
            `True` if the input is valid, `False` otherwise.
    """
    if isinstance(images, (np.ndarray, torch.Tensor)) and images.ndim == 4:
        return True
    elif is_valid_image(images):
        return True
    elif isinstance(images, list):
        return all(is_valid_image(image) for image in images)
    return False
