# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import PIL
import torch

from .image_processor import VaeImageProcessor, is_valid_image, is_valid_image_imagelist


class VideoProcessor(VaeImageProcessor):
    r"""Simple video processor."""

    def tensor2vid(
        self, video: torch.Tensor, output_type: str = "np"
    ) -> Union[np.ndarray, torch.Tensor, List[PIL.Image.Image]]:
        r"""
        Converts a video tensor to a list of frames for export.

        Args:
            video (`torch.Tensor`): The video as a tensor.
            output_type (`str`, defaults to `"np"`): Output type of the postprocessed `video` tensor.
        """
        batch_size = video.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = self.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            outputs = np.stack(outputs)
        elif output_type == "pt":
            outputs = torch.stack(outputs)
        elif not output_type == "pil":
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        return outputs

    def preprocess_video(self, video) -> torch.Tensor:
        r"""
        Preprocesses input video(s).

        Args:
            video: The input video. It can be one of the following:
                * List of the PIL images.
                * List of list of PIL images.
                * 4D Torch tensors (expected shape for each tensor: (num_frames, num_channels, height, width)).
                * 4D NumPy arrays (expected shape for each array: (num_frames, height, width, num_channels)).
                * List of 4D Torch tensors (expected shape for each tensor: (num_frames, num_channels, height, width)).
                * List of 4D NumPy arrays (expected shape for each array: (num_frames, height, width, num_channels)).
                * 5D NumPy arrays: expected shape for each array: (batch_size, num_frames, height, width,
                  num_channels).
                * 5D Torch tensors: expected shape for each array: (batch_size, num_frames, num_channels, height,
                  width).
        """
        # video processor only accepts video or a list of videos or a batch of videos (5d array/tensors) as inputs,
        # while we do accept a list of 5d array/tensors, we concatenate them to a single video batch
        if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
            video = np.concatenate(video, axis=0)
        if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
            video = torch.cat(video, axis=0)

        # ensure the input is a list of videos. if it is a batch of videos, it is converted to a list of videos
        # If it is is a single video, it is convereted to a list of one video.
        if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
            video = list(video)
        elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
            video = [video]
        elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
            video = video
        else:
            raise ValueError(
                "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
            )

        video = torch.stack([self.preprocess(img) for img in video], dim=0)
        video = video.permute(0, 2, 1, 3, 4)

        return video
