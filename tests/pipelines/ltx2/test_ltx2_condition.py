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

import torch

from diffusers import LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

from ...testing_utils import enable_full_determinism
from ..testing_utils import PipelineTesterMixin
from .testing_utils import (
    LTX2BaseTesterConfig,
    LTX2LoraMemoryTesterMixin,
    LTX2LoraTesterMixin,
    LTX2MemoryTesterMixin,
)


enable_full_determinism()


class LTX2ConditionPipelineTesterConfig(LTX2BaseTesterConfig):
    pipeline_class = LTX2ConditionPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "conditions",
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
    unset_components = ("audio_scheduler", "processor", "prompt_enhancer", "duration_head")

    def get_dummy_inputs(self):
        generator = self.get_generator(0)
        image = torch.rand((1, 3, 32, 32), generator=generator)
        # Synthetic float tensors skip H.264 CRF re-compression (training path uses PIL/uint8).
        img_cond = LTX2VideoCondition(frames=image, index=0, strength=1.0, crf=0)

        return {
            "conditions": img_cond,
            "prompt": "a robot dancing",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            # Pin legacy sampling knobs so unit tests stay stable when production defaults
            # track LTX-2.3/2.5 (STG on, modality/rescale, cross-timestep).
            "stg_scale": 0.0,
            "modality_scale": 1.0,
            "guidance_rescale": 0.0,
            "audio_guidance_scale": 1.0,
            "audio_stg_scale": 0.0,
            "audio_modality_scale": 1.0,
            "audio_guidance_rescale": 0.0,
            "spatio_temporal_guidance_blocks": None,
            "use_cross_timestep": False,
            "height": 32,
            "width": 32,
            "num_frames": 5,
            "frame_rate": 25.0,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTX2ConditionPipeline(LTX2ConditionPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=2, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestLTX2ConditionPipelineMemory(LTX2ConditionPipelineTesterConfig, LTX2MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX2 condition pipeline."""


class TestLTX2ConditionPipelineLoRA(LTX2ConditionPipelineTesterConfig, LTX2LoraTesterMixin):
    """LoRA tests for the LTX2 condition pipeline."""


class TestLTX2ConditionPipelineLoRAMemory(LTX2ConditionPipelineTesterConfig, LTX2LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LTX2 condition pipeline."""
