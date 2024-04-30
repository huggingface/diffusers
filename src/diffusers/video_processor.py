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


class VideoProcessor(VaeImageProcessor):
    r"""Simple video processor."""

    def tensor2vid(
        self, video: torch.FloatTensor, output_type: str = "np"
    ) -> Union[np.ndarray, torch.FloatTensor, List[PIL.Image.Image]]:
        r"""
        Converts a video tensor to a list of frames for export.

        Args:
            video (`torch.FloatTensor`): The video as a tensor.
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

    def preprocess_video(self, video) -> torch.FloatTensor:
        r"""
        Preprocesses input video(s).

        Args:
            video: The input video. It can be one of the following:
                * List of the PIL images.
                * List of list of PIL images.
                * List of 4D Torch tensors (expected shape for each tensor: (num_frames, num_channels, height, width)).
                * List of list of 4D Torch tensors (expected shape for tensor: (num_frames, num_channels, height,
                  width)).
                * List of 4D NumPy arrays (expected shape for each array: (num_frames, height, width, num_channels)).
                * List of list of 4D NumPy arrays (expected shape for each array: (num_frames, height, width,
                  num_channels)).
        """
        supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image)

        # Single-frame video.
        if isinstance(video, supported_formats):
            video = [video]
        elif not (isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(supported_formats)}"
            )

        # In case the video a list of list of PIL images.
        if isinstance(video, list) and isinstance(video[0], list) and isinstance(video[0][0], PIL.Image.Image):
            video_ = []
            first_video_length = 0
            for i, vid in enumerate(video):
                current_video = [np.array(frame) for frame in vid]
                if i == 0:
                    first_video_length = len(current_video)
                if len(current_video) != first_video_length:
                    raise ValueError("Cannot batch together videos of different lengths.")
                video_.append(current_video)
            video = np.stack(video_, axis=0)

        # In case the video is a list of PIL images, convert to a list of ndarrays.
        if isinstance(video[0], PIL.Image.Image):
            video = [np.array(frame) for frame in video]

        if isinstance(video[0], np.ndarray):
            # When the number of dimension of the first element in `video` is 5, it means
            # each element in the `video` list is a video.
            video = np.concatenate(video, axis=0) if video[0].ndim == 5 else np.stack(video, axis=0)

            if video.dtype == np.uint8:
                if video.min() >= 0 and video.max() <= 255:
                    raise ValueError(
                        f"The inputs don't have the correct value range for the determined data-type ({video.dtype}): {video.min()=}, {video.max()=}"
                    )
                # We perform the scaling step here so that `preprocess()` can handle things correctly for us.
                video = np.array(video).astype(np.float32) / 255.0

            if video.ndim == 4:
                video = video[None, ...]

            video = video.permute(0, 4, 1, 2, 3)

        elif isinstance(video[0], torch.Tensor):
            video = torch.cat(video, axis=0) if video[0].ndim == 5 else torch.stack(video, axis=0)

            # don't need any preprocess if the video is latents
            channel = video.shape[1]
            if channel == 4:
                return video

            # move channels before num_frames
            video = video.permute(0, 2, 1, 3, 4)

        # `preprocess()` here would return a PT tensor.
        video = torch.stack([self.preprocess(f) for f in video], dim=0)

        return video
