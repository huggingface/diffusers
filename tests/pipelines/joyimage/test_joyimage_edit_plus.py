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

from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    JoyImageEditPlusPipeline,
    JoyImageEditPlusTransformer3DModel,
)

from ...testing_utils import enable_full_determinism
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class JoyImageEditPlusPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = JoyImageEditPlusPipeline
    # `images` (a list of references per sample) takes the place of the usual single `image` input.
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "images",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    # Each sample is bound to its own set of reference images, so the pipeline generates exactly one image per
    # prompt and does not expose `num_images_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])
    batch_input_params = frozenset(["prompt", "images"])
    output_shape = (3, 32, 32)

    @pytest.fixture(autouse=True, scope="class")
    @classmethod
    def tiny_resolution_bucket(cls):
        """Pin the resolution bucket to the dummy 32x32 resolution.

        `JoyImageEditImageProcessor` resolves its working resolution through `find_best_bucket`, which only knows
        the 1024 bucket list, so an unpatched pipeline would upscale the dummy inputs to ~1024x1024.
        """
        with patch("diffusers.pipelines.joyimage.image_processor.find_best_bucket", return_value=(32, 32)):
            yield

    def get_dummy_components(self):
        tiny_ckpt_id = "huangfeice/tiny-random-Qwen3VLForConditionalGeneration"

        torch.manual_seed(0)
        transformer = JoyImageEditPlusTransformer3DModel(
            patch_size=[1, 2, 2],
            in_channels=16,
            hidden_size=32,
            num_attention_heads=2,
            text_dim=16,
            num_layers=1,
            rope_dim_list=[4, 6, 6],
            theta=256,
        )

        torch.manual_seed(0)
        vae = AutoencoderKLWan(
            base_dim=3,
            z_dim=16,
            dim_mult=[1, 1, 1, 1],
            num_res_blocks=1,
            temperal_downsample=[False, True, True],
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        processor = Qwen3VLProcessor.from_pretrained(tiny_ckpt_id)
        processor.image_processor.min_pixels = 4 * 28 * 28
        processor.image_processor.max_pixels = 4 * 28 * 28

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(tiny_ckpt_id)
        text_encoder.resize_token_embeddings(len(processor.tokenizer))

        return {
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": processor.tokenizer,
            "processor": processor,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "combine the two images",
            "images": [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))],
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestJoyImageEditPlusPipeline(JoyImageEditPlusPipelineTesterConfig, PipelineTesterMixin):
    pass


class TestJoyImageEditPlusPipelineMemory(JoyImageEditPlusPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the JoyImage Edit Plus pipeline."""
