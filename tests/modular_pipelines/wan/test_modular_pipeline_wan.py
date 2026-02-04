# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

import pytest

from diffusers.modular_pipelines import (
    WanAutoBlocks,
    WanModularPipeline,
)

from ..test_modular_pipelines_common import ModularPipelineTesterMixin


class TestWanModularPipelineFast(ModularPipelineTesterMixin):
    pipeline_class = WanModularPipeline
    pipeline_blocks_class = WanAutoBlocks
    pretrained_model_name_or_path = "hf-internal-testing/tiny-wan-modular-pipe"

    params = frozenset(["prompt", "height", "width", "num_frames"])
    batch_params = frozenset(["prompt"])
    # Override optional_params for video pipeline (uses num_videos_per_prompt instead of num_images_per_prompt)
    optional_params = frozenset(["num_inference_steps", "num_videos_per_prompt", "latents"])
    # Video pipelines output "videos" instead of "images"
    output_type = "videos"

    def get_dummy_inputs(self, seed=0):
        generator = self.get_generator(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "height": 16,
            "width": 16,
            "num_frames": 9,
            "max_sequence_length": 16,
        }
        return inputs

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-4):
        """Override to handle video-specific batching where each batch item is compared separately."""
        import torch

        from ...testing_utils import torch_device

        pipe = self.get_pipeline().to(torch_device)
        inputs = self.get_dummy_inputs()
        inputs["generator"] = self.get_generator(0)

        # batchify inputs
        batched_inputs = {}
        batched_inputs.update(inputs)

        for name in self.batch_params:
            if name not in inputs:
                continue

            value = inputs[name]
            batched_inputs[name] = batch_size * [value]

        batched_inputs["generator"] = [self.get_generator(i) for i in range(batch_size)]

        if "batch_size" in inputs:
            batched_inputs["batch_size"] = batch_size

        output = pipe(**inputs, output=self.output_type)
        output_batch = pipe(**batched_inputs, output=self.output_type)

        assert self._get_batch_size_from_output(output_batch) == batch_size

        # For video outputs, compare the first item of the batch to the single output
        output_tensor = (
            torch.from_numpy(output[0]) if isinstance(output, list) else self._convert_output_to_tensor(output)
        )
        output_batch_tensor = torch.from_numpy(output_batch[0])

        max_diff = torch.abs(output_batch_tensor - output_tensor).max()
        assert max_diff < expected_max_diff, "Batch inference results different from single inference results"

    @pytest.mark.skip(reason="Video pipelines use num_videos_per_prompt instead of num_images_per_prompt")
    def test_num_images_per_prompt(self):
        pass
