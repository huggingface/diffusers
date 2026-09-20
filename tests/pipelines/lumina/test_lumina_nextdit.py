# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, GemmaConfig, GemmaForCausalLM

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    LuminaNextDiT2DModel,
    LuminaPipeline,
)

from ...testing_utils import (
    backend_empty_cache,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


class LuminaPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LuminaPipeline
    required_input_params_in_call_signature = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_input_params = frozenset(["prompt", "negative_prompt"])
    # The dummy `AutoencoderKL` has a single block and so does not up/downsample: latents at
    # `32 // vae_scale_factor` decode to a 4x4 image rather than back to the requested 32x32.
    output_shape = (3, 4, 4)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = LuminaNextDiT2DModel(
            sample_size=4,
            patch_size=2,
            in_channels=4,
            hidden_size=4,
            num_layers=2,
            num_attention_heads=1,
            num_kv_heads=1,
            multiple_of=16,
            ffn_dim_multiplier=None,
            norm_eps=1e-5,
            learn_sigma=True,
            qk_norm=True,
            cross_attention_dim=8,
            scaling_factor=1.0,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL()

        scheduler = FlowMatchEulerDiscreteScheduler()
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/dummy-gemma")

        torch.manual_seed(0)
        config = GemmaConfig(
            head_dim=2,
            hidden_size=8,
            intermediate_size=37,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=4,
        )
        text_encoder = GemmaForCausalLM(config)

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
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestLuminaPipeline(LuminaPipelineTesterConfig, PipelineTesterMixin):
    pass


class TestLuminaPipelineMemory(LuminaPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Lumina pipeline."""


@slow
@require_torch_accelerator
class TestLuminaPipelineIntegration:
    pipeline_class = LuminaPipeline
    repo_id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, seed=0):
        return {
            "prompt": "A photo of a cat",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }

    def test_lumina_inference(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload(device=torch_device)

        image = pipe(**self.get_inputs(torch_device)).images[0]
        image_slice = image[0, :10, :10]
        expected_slice = np.array(
            [
                [0.17773438, 0.18554688, 0.22070312],
                [0.046875, 0.06640625, 0.10351562],
                [0.0, 0.0, 0.02148438],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())
        assert max_diff < 1e-4
