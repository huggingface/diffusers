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
                * List of 5D NumPy arrays (expected shape for each array: (batch_size, num_frames, height, width,
                  num_channels).
                * List of 5D Torch tensors (expected shape for each array: (batch_size, num_frames, num_channels,
                  height, width).
                * 5D NumPy arrays: expected shape for each array: (batch_size, num_frames, height, width,
                  num_channels).
                * 5D Torch tensors: expected shape for each array: (batch_size, num_frames, num_channels, height,
                  width).
        """
        supported_formats = (np.ndarray, torch.Tensor, PIL.Image.Image, list)

        # Single-frame video.
        if isinstance(video, supported_formats[:-1]):
            video = [video]

        # List of PIL images.
        elif isinstance(video, list) and isinstance(video[0], PIL.Image.Image):
            video = [video]

        elif not (isinstance(video, list) and all(isinstance(i, supported_formats) for i in video)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in video]}. Currently, we only support {', '.join(list(map(str, supported_formats)))}"
            )

        if isinstance(video[0], np.ndarray):
            # When the number of dimension of the first element in `video` is 5, it means
            # each element in the `video` list is a video.
            video = np.concatenate(video, axis=0) if video[0].ndim == 5 else np.stack(video, axis=0)

            if video.ndim == 4:
                video = video[None, ...]

        elif isinstance(video[0], torch.Tensor):
            video = torch.cat(video, dim=0) if video[0].ndim == 5 else torch.stack(video, dim=0)

            # don't need any preprocess if the video is latents
            channel = video.shape[1]
            if channel == 4:
                return video

        # List of 5d tensors/ndarrays.
        elif isinstance(video[0], list):
            if isinstance(video[0][0], (np.ndarray, torch.Tensor)):
                all_frames = []
                for list_of_videos in video:
                    temp_frames = []
                    for vid in list_of_videos:
                        if vid.ndim == 4:
                            current_vid_frames = np.stack(vid, axis=0) if isinstance(vid, np.ndarray) else vid
                        elif vid.ndim == 5:
                            current_vid_frames = (
                                np.concatenate(vid, axis=0) if isinstance(vid, np.ndarray) else torch.cat(vid, dim=0)
                            )
                        temp_frames.append(current_vid_frames)

                    # Process inner list.
                    temp_frames = (
                        np.stack(temp_frames, axis=0)
                        if isinstance(temp_frames[0], np.ndarray)
                        else torch.stack(temp_frames, axis=0)
                    )
                    all_frames.append(temp_frames)

                # Process outer list.
                video = (
                    np.concatenate(all_frames, axis=0)
                    if isinstance(all_frames[0], np.ndarray)
                    else torch.cat(all_frames, dim=0)
                )

        # `preprocess()` here would return a PT tensor.
        video = torch.stack([self.preprocess(f) for f in video], dim=0)

        # move channels before num_frames
        video = video.permute(0, 2, 1, 3, 4)

        return video
