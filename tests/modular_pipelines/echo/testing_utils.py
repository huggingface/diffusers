# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import torch

from diffusers.modular_pipelines import EchoBlocks, EchoModularPipeline

from ..testing_utils import BaseModularPipelineTesterConfig


class EchoModularPipelineTesterConfig(BaseModularPipelineTesterConfig):
    pipeline_class = EchoModularPipeline
    pipeline_blocks_class = EchoBlocks
    pretrained_model_name_or_path = "Echo-Team/tiny-echo-modular-pipe"
    params = frozenset(
        [
            "prompt",
            "image",
            "memory_images",
            "memory_audio_waveforms",
            "height",
            "width",
            "num_frames",
            "frame_rate",
            "model_frame_rate",
            "sigmas",
        ]
    )
    batch_params = frozenset(["prompt"])
    optional_params = frozenset(["num_videos_per_prompt", "latents", "audio_latents", "output_type"])
    not_params = frozenset(["negative_prompt", "guidance_scale", "num_inference_steps"])
    expected_workflow_blocks = {}
    output_name = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        return {
            "prompt": "a robot dancing",
            "image": torch.rand((1, 3, 32, 32), generator=generator),
            "memory_images": [
                torch.rand((1, 3, 32, 32), generator=generator),
                torch.rand((1, 3, 32, 32), generator=generator),
            ],
            "generator": self.get_generator(seed),
            "sigmas": [1.0, 0.0],
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            "output_type": "pt",
        }
