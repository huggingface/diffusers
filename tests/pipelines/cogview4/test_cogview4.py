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

import pytest
import torch
from transformers import AutoTokenizer, GlmConfig, GlmForCausalLM

from diffusers import AutoencoderKL, CogView4Pipeline, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler

from ...testing_utils import enable_full_determinism, require_torch_accelerator
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class CogView4PipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = CogView4Pipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = CogView4Transformer2DModel(
            patch_size=2,
            in_channels=4,
            num_layers=2,
            attention_head_dim=4,
            num_attention_heads=4,
            out_channels=4,
            text_embed_dim=32,
            time_embed_dim=8,
            condition_dim=4,
        )

        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(
            base_shift=0.25,
            max_shift=0.75,
            base_image_seq_len=256,
            use_dynamic_shifting=True,
            time_shift_type="linear",
        )

        torch.manual_seed(0)
        text_encoder_config = GlmConfig(
            hidden_size=32, intermediate_size=8, num_hidden_layers=2, num_attention_heads=4, head_dim=8
        )
        text_encoder = GlmForCausalLM(text_encoder_config)
        # TODO(aryan): change this to THUDM/CogView4 once released
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "dance monkey",
            "negative_prompt": "bad",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestCogView4Pipeline(CogView4PipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestCogView4PipelineMemory(CogView4PipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the CogView4 pipeline."""


class TestCogView4PipelineLoRA(CogView4PipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the CogView4 pipeline."""


class TestCogView4PipelineLoRAMemory(CogView4PipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the CogView4 pipeline."""

    @pytest.mark.parametrize("offload_type,use_stream", [("block_level", True), ("leaf_level", False)])
    @require_torch_accelerator
    def test_group_offloading_inference_denoiser(self, tmp_path, offload_type, use_stream):
        # TODO: We don't run the (leaf_level, True) case that is enabled for other models.
        # The reason for this can be found here: https://github.com/huggingface/diffusers/pull/11804#issuecomment-3013325338
        super().test_group_offloading_inference_denoiser(tmp_path, offload_type, use_stream)
