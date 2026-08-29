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
from transformers import AutoTokenizer, T5Config, T5EncoderModel

from diffusers import AllegroPipeline, AllegroTransformer3DModel, AutoencoderKLAllegro, DDIMScheduler

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
)


enable_full_determinism()


class AllegroPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = AllegroPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # Allegro is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )
    output_shape = (8, 3, 16, 16)

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = AllegroTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=12,
            in_channels=4,
            out_channels=4,
            num_layers=num_layers,
            cross_attention_dim=24,
            sample_width=8,
            sample_height=8,
            sample_frames=8,
            caption_channels=24,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLAllegro(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "AllegroDownBlock3D",
                "AllegroDownBlock3D",
                "AllegroDownBlock3D",
                "AllegroDownBlock3D",
            ),
            up_block_types=(
                "AllegroUpBlock3D",
                "AllegroUpBlock3D",
                "AllegroUpBlock3D",
                "AllegroUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            temporal_compression_ratio=4,
        )

        # TODO(aryan): Only for now, since VAE decoding without tiling is not yet implemented here
        vae.enable_tiling()

        torch.manual_seed(0)
        scheduler = DDIMScheduler()

        text_encoder_config = T5Config(
            **{
                "d_ff": 37,
                "d_kv": 8,
                "d_model": 24,
                "num_decoder_layers": 2,
                "num_heads": 4,
                "num_layers": 2,
                "relative_attention_num_buckets": 8,
                "vocab_size": 1103,
            }
        )
        text_encoder = T5EncoderModel(text_encoder_config)
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
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": 8,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestAllegroPipeline(AllegroPipelineTesterConfig, PipelineTesterMixin):
    # `get_dummy_components` turns tiling on because decoding without it is not yet implemented for
    # `AutoencoderKLAllegro`. Tiling is a runtime flag rather than part of the VAE config, so a reloaded pipeline
    # decodes untiled and errors out — hence the skips on the save/load round-trips below.
    @pytest.mark.skip("Decoding without tiling is not yet implemented")
    def test_save_load_local(self):
        pass

    @pytest.mark.skip("Decoding without tiling is not yet implemented")
    def test_save_load_optional_components(self):
        pass

    @pytest.mark.skip("Decoding without tiling is not yet implemented")
    def test_save_load_float16(self):
        pass

    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)


class TestAllegroPipelineMemory(AllegroPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Allegro pipeline."""

    @pytest.mark.skip("Decoding without tiling is not yet implemented")
    def test_pipeline_with_accelerator_device_map(self):
        pass


class TestAllegroPipelineCache(AllegroPipelineTesterConfig, PyramidAttentionBroadcastTesterMixin):
    """Pyramid Attention Broadcast tests for the Allegro pipeline."""


@slow
@require_torch_accelerator
class TestAllegroPipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_allegro(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = AllegroPipeline.from_pretrained("rhymes-ai/Allegro", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)
        prompt = self.prompt

        videos = pipe(
            prompt=prompt,
            height=720,
            width=1280,
            num_frames=88,
            generator=generator,
            num_inference_steps=2,
            output_type="pt",
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 88, 720, 1280, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video, expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video}"
