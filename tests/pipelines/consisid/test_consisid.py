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
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

from diffusers import AutoencoderKLCogVideoX, ConsisIDPipeline, ConsisIDTransformer3DModel, DDIMScheduler
from diffusers.utils import load_image

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class ConsisIDPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = ConsisIDPipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt", "image"])
    output_shape = (8, 3, 16, 16)
    # ConsisID is a video pipeline: it exposes `num_videos_per_prompt`, not the base default `num_images_per_prompt`.
    optional_input_params = frozenset(
        ["num_inference_steps", "num_videos_per_prompt", "generator", "latents", "output_type", "return_dict"]
    )

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = ConsisIDTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=8,
            out_channels=4,
            time_embed_dim=2,
            text_embed_dim=32,
            num_layers=1,
            sample_width=2,
            sample_height=2,
            sample_frames=9,
            patch_size=2,
            temporal_compression_ratio=4,
            max_text_seq_length=16,
            use_rotary_positional_embeddings=True,
            use_learned_positional_embeddings=True,
            cross_attn_interval=1,
            is_kps=False,
            is_train_face=True,
            cross_attn_dim_head=1,
            cross_attn_num_heads=1,
            LFE_id_dim=2,
            LFE_vit_dim=2,
            LFE_depth=5,
            LFE_dim_head=8,
            LFE_num_heads=2,
            LFE_num_id_token=1,
            LFE_num_querie=1,
            LFE_output_dim=21,
            LFE_ff_mult=1,
            LFE_num_scale=1,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLCogVideoX(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
                "CogVideoXDownBlock3D",
            ),
            up_block_types=(
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
                "CogVideoXUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            temporal_compression_ratio=4,
        )

        torch.manual_seed(0)
        scheduler = DDIMScheduler()
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

    def get_dummy_inputs(self):
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        id_vit_hidden = [torch.ones([1, 2, 2])] * 1
        id_cond = torch.ones(1, 2)
        return {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": image_height,
            "width": image_width,
            "num_frames": 8,
            "max_sequence_length": 16,
            "id_vit_hidden": id_vit_hidden,
            "id_cond": id_cond,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestConsisIDPipeline(ConsisIDPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    def test_vae_tiling(self, expected_diff_max: float = 0.4):
        components = self.get_dummy_components()

        # The reason to modify it this way is because ConsisID Transformer limits the generation to resolutions used during initialization.
        # This limitation comes from using learned positional embeddings which cannot be generated on-the-fly like sincos or RoPE embeddings.
        # See the if-statement on "self.use_learned_positional_embeddings" in diffusers/models/embeddings.py
        components["transformer"] = ConsisIDTransformer3DModel.from_config(
            components["transformer"].config,
            sample_height=16,
            sample_width=16,
        )

        pipe = self.get_pipeline(**components)

        # Without tiling
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_without_tiling = pipe(**inputs)[0]

        # With tiling
        pipe.vae.enable_tiling(
            tile_sample_min_height=96,
            tile_sample_min_width=96,
            tile_overlap_factor_height=1 / 12,
            tile_overlap_factor_width=1 / 12,
        )
        inputs = self.get_dummy_inputs()
        inputs["height"] = inputs["width"] = 128
        output_with_tiling = pipe(**inputs)[0]

        assert (output_without_tiling - output_with_tiling).max() < expected_diff_max, (
            "VAE tiling should not affect the inference results"
        )


class TestConsisIDPipelineMemory(ConsisIDPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the ConsisID pipeline."""


@slow
@require_torch_accelerator
class TestConsisIDPipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_consisid(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

        prompt = self.prompt
        image = load_image("https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/2.png?raw=true")
        id_vit_hidden = [torch.ones([1, 577, 1024])] * 5
        id_cond = torch.ones(1, 1280)

        videos = pipe(
            image=image,
            prompt=prompt,
            height=480,
            width=720,
            num_frames=16,
            id_vit_hidden=id_vit_hidden,
            id_cond=id_cond,
            generator=generator,
            num_inference_steps=1,
            output_type="pt",
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 16, 480, 720, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video.cpu(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video}"
