# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import unittest

import numpy as np
import PIL.Image
import torch
from parameterized import parameterized

from diffusers.video_processor import VideoProcessor


np.random.seed(0)
torch.manual_seed(0)


class VideoProcessorTest(unittest.TestCase):
    def get_dummy_sample(self, input_type):
        batch_size = 1
        num_frames = 5
        num_channels = 3
        height = 8
        width = 8

        def generate_image():
            return PIL.Image.fromarray(np.random.randint(0, 256, size=(height, width, num_channels)).astype("uint8"))

        def generate_4d_array():
            return np.random.rand(num_frames, height, width, num_channels)

        def generate_5d_array():
            return np.random.rand(batch_size, num_frames, height, width, num_channels)

        def generate_4d_tensor():
            return torch.rand(num_frames, num_channels, height, width)

        def generate_5d_tensor():
            return torch.rand(batch_size, num_frames, num_channels, height, width)

        if input_type == "list_images":
            sample = [generate_image() for _ in range(num_frames)]
        elif input_type == "list_list_images":
            sample = [[generate_image() for _ in range(num_frames)] for _ in range(num_frames)]
        elif input_type == "list_4d_np":
            sample = [generate_4d_array() for _ in range(num_frames)]
        elif input_type == "list_list_4d_np":
            sample = [[generate_4d_array() for _ in range(num_frames)] for _ in range(num_frames)]
        elif input_type == "list_5d_np":
            sample = [generate_5d_array() for _ in range(num_frames)]
        elif input_type == "5d_np":
            sample = generate_5d_array()
        elif input_type == "list_4d_pt":
            sample = [generate_4d_tensor() for _ in range(num_frames)]
        elif input_type == "list_list_4d_pt":
            sample = [[generate_4d_tensor() for _ in range(num_frames)] for _ in range(num_frames)]
        elif input_type == "list_5d_pt":
            sample = [generate_5d_tensor() for _ in range(num_frames)]
        elif input_type == "5d_pt":
            sample = generate_5d_tensor()

        return sample

    def to_np(self, video):
        # List of images.
        if isinstance(video[0], PIL.Image.Image):
            video = np.stack([np.array(i) for i in video], axis=0)

        # List of list of images.
        elif isinstance(video, list) and isinstance(video[0][0], PIL.Image.Image):
            frames = []
            for vid in video:
                all_current_frames = np.stack([np.array(i) for i in vid], axis=0)
                frames.append(all_current_frames)
            video = np.stack([np.array(frame) for frame in frames], axis=0)

        # List of 4d/5d {ndarrays, torch tensors}.
        elif isinstance(video, list) and isinstance(video[0], (torch.Tensor, np.ndarray)):
            if isinstance(video[0], np.ndarray):
                video = np.stack(video, axis=0) if video[0].ndim == 4 else np.concatenate(video, axis=0)
            else:
                if video[0].ndim == 4:
                    video = np.stack([i.cpu().numpy().transpose(0, 2, 3, 1) for i in video], axis=0)
                elif video[0].ndim == 5:
                    video = np.concatenate([i.cpu().numpy().transpose(0, 1, 3, 4, 2) for i in video], axis=0)

        # List of list of 4d/5d {ndarrays, torch tensors}.
        elif (
            isinstance(video, list)
            and isinstance(video[0], list)
            and isinstance(video[0][0], (torch.Tensor, np.ndarray))
        ):
            all_frames = []
            for list_of_videos in video:
                temp_frames = []
                for vid in list_of_videos:
                    if vid.ndim == 4:
                        current_vid_frames = np.stack(
                            [i if isinstance(i, np.ndarray) else i.cpu().numpy().transpose(1, 2, 0) for i in vid],
                            axis=0,
                        )
                    elif vid.ndim == 5:
                        current_vid_frames = np.concatenate(
                            [i if isinstance(i, np.ndarray) else i.cpu().numpy().transpose(0, 2, 3, 1) for i in vid],
                            axis=0,
                        )
                    temp_frames.append(current_vid_frames)
                temp_frames = np.stack(temp_frames, axis=0)
                all_frames.append(temp_frames)

            video = np.concatenate(all_frames, axis=0)

        # Just 5d {ndarrays, torch tensors}.
        elif isinstance(video, (torch.Tensor, np.ndarray)) and video.ndim == 5:
            video = video if isinstance(video, np.ndarray) else video.cpu().numpy().transpose(0, 1, 3, 4, 2)

        return video

    @parameterized.expand(["list_images", "list_list_images"])
    def test_video_processor_pil(self, input_type):
        video_processor = VideoProcessor(do_resize=False, do_normalize=True)

        input = self.get_dummy_sample(input_type=input_type)

        for output_type in ["pt", "np", "pil"]:
            out = video_processor.postprocess_video(video_processor.preprocess_video(input), output_type=output_type)
            out_np = self.to_np(out)
            input_np = self.to_np(input).astype("float32") / 255.0 if output_type != "pil" else self.to_np(input)
            assert np.abs(input_np - out_np).max() < 1e-6, f"Decoded output does not match input for {output_type=}"

    @parameterized.expand(["list_4d_np", "list_5d_np", "5d_np"])
    def test_video_processor_np(self, input_type):
        video_processor = VideoProcessor(do_resize=False, do_normalize=True)

        input = self.get_dummy_sample(input_type=input_type)

        for output_type in ["pt", "np", "pil"]:
            out = video_processor.postprocess_video(video_processor.preprocess_video(input), output_type=output_type)
            out_np = self.to_np(out)
            input_np = (
                (self.to_np(input) * 255.0).round().astype("uint8") if output_type == "pil" else self.to_np(input)
            )
            assert np.abs(input_np - out_np).max() < 1e-6, f"Decoded output does not match input for {output_type=}"

    @parameterized.expand(["list_4d_pt", "list_5d_pt", "5d_pt"])
    def test_video_processor_pt(self, input_type):
        video_processor = VideoProcessor(do_resize=False, do_normalize=True)

        input = self.get_dummy_sample(input_type=input_type)

        for output_type in ["pt", "np", "pil"]:
            out = video_processor.postprocess_video(video_processor.preprocess_video(input), output_type=output_type)
            out_np = self.to_np(out)
            input_np = (
                (self.to_np(input) * 255.0).round().astype("uint8") if output_type == "pil" else self.to_np(input)
            )
            assert np.abs(input_np - out_np).max() < 1e-6, f"Decoded output does not match input for {output_type=}"
