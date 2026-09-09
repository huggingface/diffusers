# Copyright 2025 The HuggingFace Team.
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

from contextlib import contextmanager

import torch

from diffusers import LTX2HDRPipeline
from diffusers.pipelines.ltx2 import LTX2HDRReferenceCondition

from ...testing_utils import enable_full_determinism
from ..testing_utils import PipelineTesterMixin
from .testing_utils import (
    LTX2BaseTesterConfig,
    LTX2LoraMemoryTesterMixin,
    LTX2LoraTesterMixin,
    LTX2MemoryTesterMixin,
)


enable_full_determinism()


class LTX2HDRPipelineTesterConfig(LTX2BaseTesterConfig):
    pipeline_class = LTX2HDRPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "reference_conditions",
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # `postprocess_hdr_video` permutes to channels-last for both `"pt"` and `"np"`, so this pipeline's frames come
    # back as (num_frames, height, width, channels) rather than the channels-first layout `"pt"` usually implies.
    output_shape = (5, 32, 32, 3)
    # LTX2 is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    # Unlike the other LTX2 pipelines this one renders video only, so there is no `audio_latents`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    unset_components = ("audio_scheduler",)

    def get_dummy_inputs(self):
        generator = self.get_generator(0)
        image = torch.rand((1, 3, 32, 32), generator=generator)
        img_cond = LTX2HDRReferenceCondition(frames=image, strength=1.0)

        return {
            "reference_conditions": img_cond,
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTX2HDRPipeline(LTX2HDRPipelineTesterConfig, PipelineTesterMixin):
    # The HDR video processor applies the inverse LogC3 transfer function, whose exponential blows tiny numerical
    # differences up into large pixel ones. Tests that compare two decoded runs against each other therefore
    # compare latents instead: `latent_outputs()` flips `get_dummy_inputs` over to `output_type="latent"` for the
    # duration of the base implementation.
    _force_latent_output = False

    def get_dummy_inputs(self):
        inputs = super().get_dummy_inputs()
        if self._force_latent_output:
            inputs["output_type"] = "latent"
        return inputs

    @contextmanager
    def latent_outputs(self):
        self._force_latent_output = True
        try:
            yield
        finally:
            self._force_latent_output = False

    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-4):
        with self.latent_outputs():
            super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_save_load_float16(self, tmp_path, expected_max_diff=1e-2):
        with self.latent_outputs():
            super().test_save_load_float16(tmp_path, expected_max_diff=expected_max_diff)


class TestLTX2HDRPipelineMemory(LTX2HDRPipelineTesterConfig, LTX2MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX2 HDR pipeline."""


class TestLTX2HDRPipelineLoRA(LTX2HDRPipelineTesterConfig, LTX2LoraTesterMixin):
    """LoRA tests for the LTX2 HDR pipeline."""


class TestLTX2HDRPipelineLoRAMemory(LTX2HDRPipelineTesterConfig, LTX2LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LTX2 HDR pipeline."""
