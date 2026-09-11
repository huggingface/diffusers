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

import random

import numpy as np
import pytest
import torch

from diffusers import DDIMScheduler, LDMSuperResolutionPipeline, UNet2DModel, VQModel
from diffusers.utils import PIL_INTERPOLATION

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    floats_tensor,
    load_image,
    nightly,
    require_torch,
    torch_device,
)
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LDMSuperResolutionPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LDMSuperResolutionPipeline
    required_input_params_in_call_signature = frozenset(["image", "batch_size"])
    # `__call__` takes exactly one `image` — a `PIL.Image` or a single tensor whose leading dim *is* the batch —
    # and rejects the list of images the shared batching helpers build, so there is nothing for them to batch.
    # `test_batched_image_input` below covers batching through the API the pipeline actually offers.
    batch_input_params = frozenset()
    output_shape = (3, 64, 64)
    # An unconditional upscaler: no prompt, so no `num_images_per_prompt`, and the noise is always sampled
    # internally rather than passed in as `latents`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "output_type", "return_dict"])

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=6,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        torch.manual_seed(0)
        vqvae = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
        )
        scheduler = DDIMScheduler()

        return {"unet": unet, "vqvae": vqvae, "scheduler": scheduler}

    def get_dummy_inputs(self):
        return {
            "image": floats_tensor((1, 3, 32, 32), rng=random.Random(0)),
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestLDMSuperResolutionPipeline(LDMSuperResolutionPipelineTesterConfig, PipelineTesterMixin):
    def test_inference_superresolution(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.8678, 0.8245, 0.6381, 0.6830, 0.4385, 0.5599, 0.4641, 0.6201, 0.5150])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_batched_image_input(self):
        # The `batch_size` argument is ignored for tensor input (`__call__` derives it from `image.shape[0]`), so
        # batching means stacking frames into `image` itself.
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        inputs["image"] = inputs["image"].repeat(3, 1, 1, 1)
        images = pipe(**inputs).images

        assert images.shape == (3, *self.output_shape)

    @pytest.mark.skip("`__call__` rejects a list of images; see `test_batched_image_input`.")
    def test_inference_batch_consistent(self):
        pass

    @pytest.mark.skip("`__call__` rejects a list of images; see `test_batched_image_input`.")
    def test_inference_batch_single_identical(self):
        pass


class TestLDMSuperResolutionPipelineMemory(LDMSuperResolutionPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LDM upscaler pipeline."""


@nightly
@require_torch
class TestLDMSuperResolutionPipelineIntegration:
    def test_inference_superresolution(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/vq_diffusion/teddy_bear_pool.png"
        )
        init_image = init_image.resize((64, 64), resample=PIL_INTERPOLATION["lanczos"])

        ldm = LDMSuperResolutionPipeline.from_pretrained("duongna/ldm-super-resolution")
        ldm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ldm(image=init_image, generator=generator, num_inference_steps=20, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.7644, 0.7679, 0.7642, 0.7633, 0.7666, 0.7560, 0.7425, 0.7257, 0.6907])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
