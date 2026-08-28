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

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

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


class DDPMPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = DDPMPipeline
    required_input_params_in_call_signature = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_input_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS
    # DDPM is unconditional and samples its own noise: there is no prompt to repeat
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
        scheduler = DDPMScheduler()
        return {"unet": unet, "scheduler": scheduler}

    def get_dummy_inputs(self):
        return {
            "batch_size": 1,
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            "output_type": "pt",
        }


class TestDDPMPipeline(DDPMPipelineTesterConfig, PipelineTesterMixin):
    def test_inference(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        generated_image = image[0]
        assert generated_image.shape == self.output_shape

        # fmt: off
        expected_slice = torch.tensor([0.0, 0.9996672, 0.00329116, 1.0, 0.9995991, 1.0, 0.0060907, 0.00115037, 0.0])
        # fmt: on

        # `"pt"` images are `(channels, height, width)`, so the trailing-channel corner slice of the old
        # `"np"` layout is the last channel's bottom-right 3x3 block here.
        generated_slice = generated_image[-1, -3:, -3:].flatten()
        assert_tensors_close(generated_slice, expected_slice, atol=1e-2)

    def test_inference_predict_sample(self):
        # `prediction_type="sample"` makes the UNet output the denoised sample rather than the noise, so the
        # scheduler consumes it differently and the pipeline must produce a different image than the default
        # `epsilon` parameterization.
        pipe = self.get_pipeline().to(torch_device)
        output_epsilon = self.run_pipe(pipe)

        components = self.get_dummy_components()
        components["scheduler"] = DDPMScheduler(prediction_type="sample")
        pipe_sample = self.get_pipeline(**components).to(torch_device)
        output_sample = self.run_pipe(pipe_sample)

        assert output_sample.shape == output_epsilon.shape
        assert not torch.isnan(output_sample).any()
        assert not torch.allclose(output_sample, output_epsilon, atol=1e-3)


class TestDDPMPipelineMemory(DDPMPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the DDPM pipeline."""


@slow
@require_torch_accelerator
class TestDDPMPipelineIntegration:
    def test_inference_cifar10(self):
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        scheduler = DDPMScheduler.from_pretrained(model_id)

        ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        image = ddpm(generator=generator, output_type="np").images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4200, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647, 0.4155, 0.3582, 0.3385])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
