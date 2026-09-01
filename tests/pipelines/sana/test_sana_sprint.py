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
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizer

from diffusers import AutoencoderDC, SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler

from ...testing_utils import IS_GITHUB_ACTIONS, enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class SanaSprintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = SanaSprintPipeline
    # Sana Sprint is a timestep-distilled model: it takes no negative prompt.
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt"])
    output_shape = (3, 32, 32)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = SanaTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            sample_size=32,
            qk_norm="rms_norm_across_heads",
            guidance_embeds=True,
        )

        torch.manual_seed(0)
        vae = AutoencoderDC(
            in_channels=3,
            latent_channels=4,
            attention_head_dim=2,
            encoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            decoder_block_types=(
                "ResBlock",
                "EfficientViTBlock",
            ),
            encoder_block_out_channels=(8, 8),
            decoder_block_out_channels=(8, 8),
            encoder_qkv_multiscales=((), (5,)),
            decoder_qkv_multiscales=((), (5,)),
            encoder_layers_per_block=(1, 1),
            decoder_layers_per_block=[1, 1],
            downsample_block_type="conv",
            upsample_block_type="interpolate",
            decoder_norm_types="rms_norm",
            decoder_act_fns="silu",
            scaling_factor=0.41407,
        )

        torch.manual_seed(0)
        scheduler = SCMScheduler()

        torch.manual_seed(0)
        text_encoder_config = Gemma2Config(
            head_dim=16,
            hidden_size=8,
            initializer_range=0.02,
            intermediate_size=64,
            max_position_embeddings=8192,
            model_type="gemma2",
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=2,
            vocab_size=8,
            attn_implementation="eager",
        )
        text_encoder = Gemma2Model(text_encoder_config)
        tokenizer = GemmaTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
            "complex_human_instruction": None,
        }


class TestSanaSprintPipeline(SanaSprintPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        image = pipe(**self.get_dummy_inputs())[0]
        generated_image = image[0]

        assert generated_image.shape == self.output_shape

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        pipe = self.get_pipeline().to(torch_device)

        # Without tiling
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )

    # `SanaSprintPipeline` recomputes `latents` from `denoised` after the denoising loop
    # (`latents = denoised / sigma_data`), so a callback that overwrites `latents` on the last step cannot
    # influence the output. Run the base test's callback-contract checks and drop its final zeroing assertion.
    def test_callback_inputs(self):
        pipe = self.get_pipeline().to(torch_device)
        assert hasattr(pipe, "_callback_tensor_inputs"), (
            f"{self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables "
            "its callback function can use as inputs"
        )

        def callback_inputs_subset(pipe, i, t, callback_kwargs):
            for tensor_name in callback_kwargs:
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        def callback_inputs_all(pipe, i, t, callback_kwargs):
            for tensor_name in pipe._callback_tensor_inputs:
                assert tensor_name in callback_kwargs
            for tensor_name in callback_kwargs:
                assert tensor_name in pipe._callback_tensor_inputs
            return callback_kwargs

        inputs = self.get_dummy_inputs()

        # Test passing in a subset
        inputs["callback_on_step_end"] = callback_inputs_subset
        inputs["callback_on_step_end_tensor_inputs"] = ["latents"]
        inputs["output_type"] = "latent"
        _ = pipe(**inputs)[0]

        # Test passing in everything
        inputs["callback_on_step_end"] = callback_inputs_all
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["output_type"] = "latent"
        _ = pipe(**inputs)[0]

    # TODO(aryan): Create a dummy gemma model with smol vocab size
    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip(
        "A very small vocab size is used for fast tests. So, Any kind of prompt other than the empty default used in other tests will lead to a embedding lookup error. This test uses a long prompt that causes the error."
    )
    def test_inference_batch_single_identical(self):
        pass


class TestSanaSprintPipelineMemory(SanaSprintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Sana Sprint pipeline."""

    @pytest.mark.skipif(IS_GITHUB_ACTIONS, reason="Skipping test inside GitHub Actions environment")
    def test_layerwise_casting_inference(self):
        super().test_layerwise_casting_inference()
