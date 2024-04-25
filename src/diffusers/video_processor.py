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

from .image_processor import VaeImageProcessor


class VideoProcessor:
    """Simple video processor."""

    @staticmethod
    def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
        """Converts a video tensor to a list of frames for export."""
        batch_size, channels, num_frames, height, width = video.shape
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = processor.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            outputs = np.stack(outputs)
        elif output_type == "pt":
            outputs = torch.stack(outputs)
        elif not output_type == "pil":
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        return outputs

    @staticmethod
    def preprocess_video(video: List[Union[PIL.Image.Image, np.ndarray, torch.Tensor]]):
        """Preprocesses an input video."""
        supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image)

        if isinstance(video, supported_formats):
            video = [video]
        elif not (isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(video[0], PIL.Image.Image):
            video = [np.array(frame) for frame in video]

        if isinstance(video[0], np.ndarray):
            video = np.concatenate(video, axis=0) if video[0].ndim == 5 else np.stack(video, axis=0)

            # Notes from (sayakpaul): do we want to follow something similar to VaeImageProcessor here i.e.,
            # have methods `normalize()` and `denormalize()`?
            if video.dtype == np.uint8:
                video = np.array(video).astype(np.float32) / 255.0

            if video.ndim == 4:
                video = video[None, ...]

            video = torch.from_numpy(video.transpose(0, 4, 1, 2, 3))

        elif isinstance(video[0], torch.Tensor):
            video = torch.cat(video, axis=0) if video[0].ndim == 5 else torch.stack(video, axis=0)

            # don't need any preprocess if the video is latents
            channel = video.shape[1]
            if channel == 4:
                return video

            # move channels before num_frames
            video = video.permute(0, 2, 1, 3, 4)

        # normalize video
        video = 2.0 * video - 1.0

        return video
