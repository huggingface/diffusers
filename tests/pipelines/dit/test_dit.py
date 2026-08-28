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

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DiTTransformer2DModel, DPMSolverMultistepScheduler

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    load_numpy,
    nightly,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class DiTPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = DiTPipeline
    required_input_params_in_call_signature = CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS
    batch_input_params = CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS
    # DiT is class-conditioned and samples its own noise: there is no prompt to repeat
    # (`num_images_per_prompt`) and no user-suppliable `latents`.
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {"num_images_per_prompt", "latents"}
    output_shape = (3, 16, 16)

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = DiTTransformer2DModel(
            sample_size=16,
            num_layers=2,
            patch_size=4,
            attention_head_dim=8,
            num_attention_heads=2,
            in_channels=4,
            out_channels=8,
            attention_bias=True,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=1000,
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
        )
        vae = AutoencoderKL()
        scheduler = DDIMScheduler()
        return {"transformer": transformer, "vae": vae, "scheduler": scheduler}

    def get_dummy_inputs(self):
        return {
            "class_labels": [1],
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestDiTPipeline(DiTPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.2946, 0.6601, 0.4329, 0.3296, 0.4144, 0.5319, 0.7273, 0.5013, 0.4457])
        # fmt: on

        # `"pt"` images are `(channels, height, width)`, so the trailing-channel corner slice of the old
        # `"np"` layout is the last channel's bottom-right 3x3 block here.
        generated_slice = generated_image[-1, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=1e-3)


class TestDiTPipelineMemory(DiTPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the DiT pipeline."""


@nightly
@require_torch_accelerator
class TestDiTPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_dit_256(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")
        pipe.to(torch_device)

        words = ["vase", "umbrella", "white shark", "white wolf"]
        ids = pipe.get_label_ids(words)

        images = pipe(ids, generator=generator, num_inference_steps=40, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}.npy"
            )
            assert np.abs((expected_image - image).max()) < 1e-2

    def test_dit_512(self):
        pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-512")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to(torch_device)

        words = ["vase", "umbrella"]
        ids = pipe.get_label_ids(words)

        generator = torch.manual_seed(0)
        images = pipe(ids, generator=generator, num_inference_steps=25, output_type="np").images

        for word, image in zip(words, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/{word}_512.npy"
            )

            expected_slice = expected_image.flatten()
            output_slice = image.flatten()

            assert numpy_cosine_similarity_distance(expected_slice, output_slice) < 1e-2
