# coding=utf-8
# Copyright 2025 Latte Team and HuggingFace Inc.
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

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LattePipeline,
    LatteTransformer3DModel,
)

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    FasterCacheTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    PyramidAttentionBroadcastTesterMixin,
)


enable_full_determinism()


class LattePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LattePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (1, 3, 8, 8)

    def get_dummy_components(self, num_layers: int = 1):
        torch.manual_seed(0)
        transformer = LatteTransformer3DModel(
            sample_size=8,
            num_layers=num_layers,
            patch_size=2,
            attention_head_dim=8,
            num_attention_heads=3,
            caption_channels=32,
            in_channels=4,
            cross_attention_dim=24,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_single",
            norm_elementwise_affine=False,
            norm_eps=1e-6,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = DDIMScheduler()
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
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "low quality",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "video_length": 1,
            "clean_caption": False,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLattePipeline(LattePipelineTesterConfig, PipelineTesterMixin):
    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=1e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("`encode_prompt()` has multiple returns, which the shared test cannot unpack.")
    def test_encode_prompt_works_in_isolation(self):
        pass

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1.0):
        # `tokenizer` and `text_encoder` are the optional components here, so the pipeline can no longer turn a
        # prompt into embeddings once they are dropped. Encode up front and drive the reloaded pipeline with the
        # embeddings instead of the shared implementation's plain `get_dummy_inputs()`.
        pipe = self.get_pipeline().to(torch_device)

        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(self.get_dummy_inputs()["prompt"])

        inputs = self.get_dummy_inputs()
        inputs.pop("prompt")
        inputs.update(
            {
                "prompt_embeds": prompt_embeds,
                "negative_prompt": None,
                "negative_prompt_embeds": negative_prompt_embeds,
                "mask_feature": False,
            }
        )

        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)

        output = pipe(**inputs)[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in pipe._optional_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        output_loaded = pipe_loaded(**inputs)[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )


class TestLattePipelineMemory(LattePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Latte pipeline."""


class TestLattePipelinePyramidAttentionBroadcast(LattePipelineTesterConfig, PyramidAttentionBroadcastTesterMixin):
    """Pyramid Attention Broadcast tests for the Latte pipeline."""

    PAB_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "temporal_attention_block_skip_range": 2,
        "cross_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (100, 700),
        "temporal_attention_timestep_skip_range": (100, 800),
        "cross_attention_timestep_skip_range": (100, 800),
        "spatial_attention_block_identifiers": ["transformer_blocks"],
        "temporal_attention_block_identifiers": ["temporal_transformer_blocks"],
        "cross_attention_block_identifiers": ["transformer_blocks"],
    }


class TestLattePipelineFasterCache(LattePipelineTesterConfig, FasterCacheTesterMixin):
    """FasterCache tests for the Latte pipeline."""

    FASTER_CACHE_CONFIG = {
        "spatial_attention_block_skip_range": 2,
        "temporal_attention_block_skip_range": 2,
        "spatial_attention_timestep_skip_range": (-1, 901),
        "temporal_attention_timestep_skip_range": (-1, 901),
        "unconditional_batch_skip_range": 2,
        "attention_weight_callback": lambda _: 0.5,
    }


@slow
@require_torch_accelerator
class TestLattePipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_latte(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = LattePipeline.from_pretrained("maxin-cn/Latte-1", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload(device=torch_device)

        videos = pipe(
            prompt=self.prompt,
            height=512,
            width=512,
            generator=generator,
            num_inference_steps=2,
            clean_caption=False,
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 512, 512, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video.flatten(), expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video.flatten()}"
