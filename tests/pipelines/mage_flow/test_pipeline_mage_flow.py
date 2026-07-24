# Copyright 2025 The Mage Team and The HuggingFace Team. All rights reserved.
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
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen2Tokenizer

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    MageFlowPipeline,
)
from diffusers.models.autoencoders.autoencoder_mage_vae import AutoencoderMageVAE
from diffusers.models.transformers.transformer_mage_flow import MageFlowTransformer2DModel

from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class MageFlowPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MageFlowPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = MageFlowTransformer2DModel(
            in_channels=4,
            out_channels=4,
            context_in_dim=32,
            hidden_size=32,
            num_attention_heads=2,
            num_layers=1,
            axes_dim=[4, 6, 6],
            patch_size=1,
        )

        torch.manual_seed(0)
        vae = AutoencoderMageVAE(
            latent_channels=4,
            downsample_factor=16,
            encoder_hidden_size=16,
            encoder_num_blocks=1,
            encoder_patch_size=16,
            encoder_head_size=32,
            encoder_num_head_blocks=1,
            decoder_hidden_size=32,
            decoder_hidden_size_x=8,
            decoder_num_blocks=2,
            decoder_num_cond_blocks=1,
            decoder_bottleneck_dim=4,
            decoder_patch_size=16,
            sample_posterior=False,
        )

        scheduler = FlowMatchEulerDiscreteScheduler(shift=6.0)

        torch.manual_seed(0)
        config = Qwen3VLConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
            },
            hidden_size=32,
            vocab_size=152064,
        )
        text_encoder = Qwen3VLForConditionalGeneration(config).eval()

        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A cat",
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "pt",
            "generator": self.get_generator(0),
        }


class TestMageFlowPipeline(MageFlowPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-2):
        # Higher tolerance: Qwen-based text encoders with padding-sensitive attention produce
        # slightly different embeddings when batch sizes differ, amplified by CFG.
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestMageFlowPipelineMemory(MageFlowPipelineTesterConfig, MemoryTesterMixin):
    pass
