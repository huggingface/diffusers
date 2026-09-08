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
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import (
    AutoencoderKLLTXVideo,
    FlowMatchEulerDiscreteScheduler,
    LTXConditionPipeline,
    LTXVideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

from ...testing_utils import assert_tensors_close, enable_full_determinism, torch_device
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class LTXConditionPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LTXConditionPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "conditions",
            "image",
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])
    output_shape = (9, 3, 32, 32)
    # LTX is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LTXVideoTransformer3DModel(
            in_channels=8,
            out_channels=8,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=4,
            attention_head_dim=8,
            cross_attention_dim=32,
            num_layers=1,
            caption_channels=32,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLLTXVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=8,
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=(1, 1, 1, 1, 1),
            decoder_layers_per_block=(1, 1, 1, 1, 1),
            spatio_temporal_scaling=(True, True, False, False),
            decoder_spatio_temporal_scaling=(True, True, False, False),
            decoder_inject_noise=(False, False, False, False, False),
            upsample_residual=(False, False, False, False),
            upsample_factor=(1, 1, 1, 1),
            timestep_conditioning=False,
            patch_size=1,
            patch_size_t=1,
            encoder_causal=True,
            decoder_causal=False,
        )
        vae.use_framewise_encoding = False
        vae.use_framewise_decoding = False

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        # `eval()` because a directly constructed model stays in training mode, which leaves T5's
        # dropout active and makes the pipeline outputs non-deterministic across calls.
        text_encoder = T5EncoderModel(config).eval()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self, use_conditions: bool = False):
        """Dummy inputs for the pipeline.

        The conditioning image can be passed either directly as `image` or wrapped in an `LTXVideoCondition`;
        `use_conditions` picks the latter so `test_inference` can check the two routes agree.
        """
        generator = self.get_generator(0)
        image = torch.randn((1, 3, 32, 32), generator=generator)

        return {
            "conditions": LTXVideoCondition(image=image) if use_conditions else None,
            "image": None if use_conditions else image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLTXConditionPipeline(LTXConditionPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline().to(torch_device)

        generated_video = pipe(**self.get_dummy_inputs()).frames[0]
        generated_video_with_conditions = pipe(**self.get_dummy_inputs(use_conditions=True)).frames[0]

        assert generated_video.shape == self.output_shape
        assert_tensors_close(
            generated_video_with_conditions,
            generated_video,
            atol=1e-3,
            msg="Passing the image through `conditions` should match passing it as `image`.",
        )

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

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


class TestLTXConditionPipelineMemory(LTXConditionPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LTX condition pipeline."""


class TestLTXConditionPipelineLoRA(LTXConditionPipelineTesterConfig, LoraTesterMixin):
    """LoRA tests for the LTX condition pipeline."""


class TestLTXConditionPipelineLoRAMemory(LTXConditionPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x memory-optimization tests (group offload, CPU offload) for the LTX condition pipeline."""
