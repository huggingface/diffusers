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
from transformers import Qwen2Tokenizer, Qwen2VLForConditionalGeneration

from diffusers import (
    AutoencoderKLMagvit,
    EasyAnimatePipeline,
    EasyAnimateTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)

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


class EasyAnimatePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = EasyAnimatePipeline
    required_input_params_in_call_signature = frozenset(
        ["prompt", "height", "width", "guidance_scale", "negative_prompt", "prompt_embeds", "negative_prompt_embeds"]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    output_shape = (5, 3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = EasyAnimateTransformer3DModel(
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=4,
            out_channels=4,
            time_embed_dim=2,
            text_embed_dim=16,  # Must match with tiny-random-t5
            num_layers=1,
            sample_width=16,  # latent width: 2 -> final width: 16
            sample_height=16,  # latent height: 2 -> final height: 16
            patch_size=2,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLMagvit(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "SpatialDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
                "SpatialTemporalDownBlock3D",
            ),
            up_block_types=(
                "SpatialUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
                "SpatialTemporalUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            latent_channels=4,
            layers_per_block=1,
            norm_num_groups=2,
            spatial_group_norm=False,
        )

        torch.manual_seed(0)
        scheduler = FlowMatchEulerDiscreteScheduler()
        text_encoder = Qwen2VLForConditionalGeneration.from_pretrained(
            "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
        )
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
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": 5,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestEasyAnimatePipeline(EasyAnimatePipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        pipe = self.get_pipeline()

        video = pipe(**self.get_dummy_inputs()).frames
        generated_video = video[0]

        assert generated_video.shape == self.output_shape


class TestEasyAnimatePipelineMemory(EasyAnimatePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the EasyAnimate pipeline."""

    # Sequential offload of the `Qwen2VLForConditionalGeneration` text encoder leaves an int64 buffer on the meta
    # device, which accelerate's onload hook then fails to copy ("Cannot copy out of meta tensor; no data!").
    # Pre-existing, unrelated to the pipeline itself; model CPU offload and group offload are unaffected.
    SEQUENTIAL_OFFLOAD_XFAIL = pytest.mark.xfail(
        condition=True,
        reason="Sequential CPU offload leaves a meta-device buffer in the Qwen2-VL text encoder.",
        strict=False,
    )

    @SEQUENTIAL_OFFLOAD_XFAIL
    def test_sequential_cpu_offload_forward_pass(self, base_pipe_output, expected_max_diff=1e-4):
        super().test_sequential_cpu_offload_forward_pass(base_pipe_output, expected_max_diff=expected_max_diff)

    @SEQUENTIAL_OFFLOAD_XFAIL
    def test_sequential_offload_forward_pass_twice(self, expected_max_diff=2e-4):
        super().test_sequential_offload_forward_pass_twice(expected_max_diff=expected_max_diff)


@slow
@require_torch_accelerator
class TestEasyAnimatePipelineIntegration:
    prompt = "A painting of a squirrel eating a burger."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_EasyAnimate(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = EasyAnimatePipeline.from_pretrained("alibaba-pai/EasyAnimateV5.1-12b-zh", torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()
        prompt = self.prompt

        videos = pipe(
            prompt=prompt,
            height=480,
            width=720,
            num_frames=5,
            generator=generator,
            num_inference_steps=2,
            output_type="pt",
        ).frames

        video = videos[0]
        expected_video = torch.randn(1, 5, 480, 720, 3).numpy()

        max_diff = numpy_cosine_similarity_distance(video, expected_video)
        assert max_diff < 1e-3, f"Max diff is too high. got {video}"
