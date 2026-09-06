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
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    T5Config,
    T5EncoderModel,
)

from diffusers import (
    AdaptiveProjectedMixGuidance,
    AutoencoderKLHunyuanImage,
    FlowMatchEulerDiscreteScheduler,
    HunyuanImagePipeline,
    HunyuanImageTransformer2DModel,
)

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import (
    BasePipelineTesterConfig,
    FirstBlockCacheTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class HunyuanImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = HunyuanImagePipeline
    required_input_params_in_call_signature = frozenset(["prompt", "height", "width"])
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (3, 16, 16)

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1, guidance_embeds: bool = False):
        torch.manual_seed(0)
        transformer = HunyuanImageTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=4,
            attention_head_dim=8,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            num_refiner_layers=1,
            patch_size=(1, 1),
            guidance_embeds=guidance_embeds,
            text_embed_dim=32,
            text_embed_2_dim=32,
            rope_axes_dim=(4, 4),
        )

        torch.manual_seed(0)
        vae = AutoencoderKLHunyuanImage(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            block_out_channels=(32, 64, 64, 64),
            layers_per_block=1,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            sample_size=128,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

        if not guidance_embeds:
            torch.manual_seed(0)
            guider = AdaptiveProjectedMixGuidance(adaptive_projected_guidance_start_step=2)
            ocr_guider = AdaptiveProjectedMixGuidance(adaptive_projected_guidance_start_step=3)
        else:
            guider = None
            ocr_guider = None
        torch.manual_seed(0)
        config = Qwen2_5_VLConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rope_scaling": {
                    "mrope_section": [2, 2, 4],
                    "rope_type": "default",
                    "type": "default",
                },
                "rope_theta": 1000000.0,
            },
            vision_config={
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_heads": 2,
                "out_hidden_size": 32,
            },
            hidden_size=32,
            vocab_size=152064,
            vision_end_token_id=151653,
            vision_start_token_id=151652,
            vision_token_id=151654,
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration(config)
        tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")

        torch.manual_seed(0)
        t5_config = T5Config(
            d_model=32,
            d_kv=4,
            d_ff=16,
            num_layers=2,
            num_heads=2,
            relative_attention_num_buckets=8,
            relative_attention_max_distance=32,
            vocab_size=256,
            feed_forward_proj="gated-gelu",
            dense_act_fn="gelu_new",
            is_encoder_decoder=False,
            use_cache=False,
            tie_word_embeddings=False,
        )
        text_encoder_2 = T5EncoderModel(t5_config)
        tokenizer_2 = ByT5Tokenizer()

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "guider": guider,
            "ocr_guider": ocr_guider,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 5,
            "height": 16,
            "width": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestHunyuanImagePipeline(HunyuanImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.6252659, 0.51482046, 0.60799813, 0.59267783, 0.488082, 0.5857634, 0.523781, 0.58028054, 0.5674121])
        # fmt: on

        generated_slice = generated_image[0, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_guider(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        pipe.guider = pipe.guider.new(guidance_scale=1000)
        pipe.ocr_guider = pipe.ocr_guider.new(guidance_scale=1000)

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.6068114, 0.48716035, 0.5984431, 0.60241306, 0.48849544, 0.5624479, 0.53696984, 0.58964247, 0.54248774])
        # fmt: on

        generated_slice = generated_image[0, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_with_distilled_guidance(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline(**self.get_dummy_components(guidance_embeds=True))

        image = pipe(**self.get_dummy_inputs(), distilled_guidance_scale=3.5).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.63667065, 0.5187377, 0.66757566, 0.6320319, 0.4913387, 0.54813194, 0.5335031, 0.5736143, 0.5461346])
        # fmt: on

        generated_slice = generated_image[0, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        pipe = self.get_pipeline()

        # Without tiling
        output_without_tiling = self.run_pipe(pipe, height=128, width=128)

        # With tiling
        pipe.vae.enable_tiling(tile_sample_min_size=96)
        output_with_tiling = self.run_pipe(pipe, height=128, width=128)

        assert_tensors_close(
            output_with_tiling,
            output_without_tiling,
            atol=expected_diff_max,
            msg="VAE tiling should not affect the inference results.",
        )

    @pytest.mark.skip("TODO: Test not supported for now because needs to be adjusted to work with guiders.")
    def test_encode_prompt_works_in_isolation(self):
        pass


class TestHunyuanImagePipelineMemory(HunyuanImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the HunyuanImage pipeline."""


class TestHunyuanImagePipelineFirstBlockCache(HunyuanImagePipelineTesterConfig, FirstBlockCacheTesterMixin):
    """First Block Cache tests for the HunyuanImage pipeline."""
