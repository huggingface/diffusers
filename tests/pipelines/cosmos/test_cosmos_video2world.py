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

import PIL.Image
import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLCosmos, CosmosTransformer3DModel, CosmosVideoToWorldPipeline, EDMEulerScheduler

from ...testing_utils import assert_tensors_close, enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin
from .cosmos_guardrail import DummyCosmosSafetyChecker
from .testing_utils import CosmosSafetyCheckerTesterMixin


enable_full_determinism()


class CosmosVideoToWorldPipelineWrapper(CosmosVideoToWorldPipeline):
    @staticmethod
    def from_pretrained(*args, **kwargs):
        kwargs["safety_checker"] = DummyCosmosSafetyChecker()
        return CosmosVideoToWorldPipeline.from_pretrained(*args, **kwargs)


class CosmosVideoToWorldPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = CosmosVideoToWorldPipelineWrapper
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image", "video"])
    output_shape = (9, 3, 32, 32)
    # Cosmos text-to-world is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = CosmosTransformer3DModel(
            in_channels=4 + 1,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            mlp_ratio=2,
            text_embed_dim=32,
            adaln_lora_dim=4,
            max_size=(4, 32, 32),
            patch_size=(1, 2, 2),
            rope_scale=(2.0, 1.0, 1.0),
            concat_padding_mask=True,
            extra_pos_embed_type="learnable",
        )

        torch.manual_seed(0)
        vae = AutoencoderKLCosmos(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            encoder_block_out_channels=(8, 8, 8, 8),
            decode_block_out_channels=(8, 8, 8, 8),
            attention_resolutions=(8,),
            resolution=64,
            num_layers=2,
            patch_size=4,
            patch_type="haar",
            scaling_factor=1.0,
            spatial_compression_ratio=4,
            temporal_compression_ratio=4,
        )

        torch.manual_seed(0)
        scheduler = EDMEulerScheduler(
            sigma_min=0.002,
            sigma_max=80,
            sigma_data=0.5,
            sigma_schedule="karras",
            num_train_timesteps=1000,
            prediction_type="epsilon",
            rho=7.0,
            final_sigmas_type="sigma_min",
        )
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
            # We cannot run the Cosmos Guardrail for fast tests due to the large model size
            "safety_checker": DummyCosmosSafetyChecker(),
        }

    def get_dummy_inputs(self):
        image_height = 32
        image_width = 32
        image = PIL.Image.new("RGB", (image_width, image_height))
        return {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": image_height,
            "width": image_width,
            "num_frames": 9,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestCosmosVideoToWorldPipeline(
    CosmosVideoToWorldPipelineTesterConfig, CosmosSafetyCheckerTesterMixin, PipelineTesterMixin
):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]
        assert generated_video.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.0, 0.8275, 0.7529, 0.7294, 0.0, 0.6, 1.0, 0.3804, 0.6667, 0.0863, 0.8784, 0.5922, 0.6627, 0.2784, 0.5725, 0.7765])
        # fmt: on

        generated_slice = generated_video.flatten()
        generated_slice = torch.cat([generated_slice[:8], generated_slice[-8:]])
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-2):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        pipe = self.get_pipeline()

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

        assert (output_without_tiling - output_with_tiling).max() < expected_diff_max, (
            "VAE tiling should not affect the inference results"
        )


class TestCosmosVideoToWorldPipelineMemory(CosmosVideoToWorldPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Cosmos video-to-world pipeline."""
