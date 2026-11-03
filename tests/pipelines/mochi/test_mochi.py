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

import gc

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLMochi, FlowMatchEulerDiscreteScheduler, MochiPipeline, MochiTransformer3DModel

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    nightly,
    numpy_cosine_similarity_distance,
    require_big_accelerator,
    require_torch_accelerator,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FasterCacheTesterMixin,
    FirstBlockCacheTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


enable_full_determinism()


class MochiPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = MochiPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (7, 3, 16, 16)
    # Mochi is a video pipeline: it exposes `num_videos_per_prompt`, not `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self, num_layers: int = 2):
        torch.manual_seed(0)
        transformer = MochiTransformer3DModel(
            patch_size=2,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=num_layers,
            pooled_projection_dim=16,
            in_channels=12,
            out_channels=None,
            qk_norm="rms_norm",
            text_embed_dim=32,
            time_embed_dim=4,
            activation_fn="swiglu",
            max_sequence_length=16,
        )
        transformer.pos_frequencies.data = transformer.pos_frequencies.new_full(transformer.pos_frequencies.shape, 0)

        torch.manual_seed(0)
        vae = AutoencoderKLMochi(
            latent_channels=12,
            out_channels=3,
            encoder_block_out_channels=(32, 32, 32, 32),
            decoder_block_out_channels=(32, 32, 32, 32),
            layers_per_block=(1, 1, 1, 1, 1),
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
        text_encoder = T5EncoderModel(config)
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

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
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": 16,
            "width": 16,
            # 6 * k + 1 is the recommendation
            "num_frames": 7,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestMochiPipeline(MochiPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_vae_tiling(self, expected_diff_max: float = 0.2):
        pipe = self.get_pipeline().to(torch_device)

        # Without tiling
        output_without_tiling = self.run_pipe(pipe, height=128, width=128)

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_sample_stride_height=64,
            tile_sample_stride_width=64,
        )
        output_with_tiling = self.run_pipe(pipe, height=128, width=128)

        assert (output_without_tiling - output_with_tiling).abs().max() < expected_diff_max, (
            "VAE tiling should not affect the inference results."
        )


class TestMochiPipelineMemory(MochiPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Mochi pipeline."""


class TestMochiPipelineFasterCache(MochiPipelineTesterConfig, FasterCacheTesterMixin):
    """FasterCache tests for the Mochi pipeline."""


class TestMochiPipelineFirstBlockCache(MochiPipelineTesterConfig, FirstBlockCacheTesterMixin):
    """First Block Cache tests for the Mochi pipeline."""


@nightly
@require_torch_accelerator
@require_big_accelerator
class TestMochiPipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_mochi(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        videos = pipe(
            prompt=prompt,
            height=480,
            width=848,
            num_frames=19,
            generator=generator,
            num_inference_steps=2,
            output_type="pt",
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 19, 480, 848, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video.cpu(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video}"
