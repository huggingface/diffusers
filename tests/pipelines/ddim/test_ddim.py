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

import numpy as np
import torch

from diffusers import DDIMPipeline, DDIMScheduler, UNet2DModel

from ...testing_utils import (
    assert_tensors_close,
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS, UNCONDITIONAL_IMAGE_GENERATION_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class DDIMPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = DDIMPipeline
    required_input_params_in_call_signature = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_input_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS
    # DDIM is unconditional and samples its own noise: there is no prompt to repeat
    # (`num_images_per_prompt`) and no user-suppliable `latents`.
    optional_input_params = BasePipelineTesterConfig.optional_input_params - {"num_images_per_prompt", "latents"}
    output_shape = (3, 8, 8)

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            norm_num_groups=4,
            sample_size=8,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = DDIMScheduler()
        return {"unet": unet, "scheduler": scheduler}

    def get_dummy_inputs(self):
        return {
            "batch_size": 1,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestDDIMPipeline(DDIMPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.0, 9.979e-01, 0.0, 9.999e-01, 9.986e-01, 9.991e-01, 7.106e-04, 0.0, 0.0])
        # fmt: on

        # `"pt"` images are `(channels, height, width)`, so the trailing-channel corner slice of the old
        # `"np"` layout is the last channel's bottom-right 3x3 block here.
        generated_slice = generated_image[-1, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-3)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=3e-3)

    def test_save_load_local(self, tmp_path, base_pipe_output):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


class TestDDIMPipelineMemory(DDIMPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the DDIM pipeline."""


@slow
@require_torch_accelerator
class TestDDIMPipelineIntegration:
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler()

        ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddim(generator=generator, eta=0.0, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.1723, 0.1617, 0.1600, 0.1626, 0.1497, 0.1513, 0.1505, 0.1442, 0.1453])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_ema_bedroom(self):
        model_id = "google/ddpm-ema-bedroom-256"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDIMScheduler.from_pretrained(model_id)

        ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0060, 0.0201, 0.0344, 0.0024, 0.0018, 0.0002, 0.0022, 0.0000, 0.0069])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
