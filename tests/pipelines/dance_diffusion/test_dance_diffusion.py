# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import unittest

import numpy as np
import torch

from diffusers import DanceDiffusionPipeline, IPNDMScheduler, UNet1DModel
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu, skip_mps

from ..pipeline_params import UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS, UNCONDITIONAL_AUDIO_GENERATION_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class DanceDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = DanceDiffusionPipeline
    params = UNCONDITIONAL_AUDIO_GENERATION_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {
        "callback",
        "latents",
        "callback_steps",
        "output_type",
        "num_images_per_prompt",
    }
    batch_params = UNCONDITIONAL_AUDIO_GENERATION_BATCH_PARAMS
    test_attention_slicing = False
    test_cpu_offload = False

    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet1DModel(
            block_out_channels=(32, 32, 64),
            extra_in_channels=16,
            sample_size=512,
            sample_rate=16_000,
            in_channels=2,
            out_channels=2,
            flip_sin_to_cos=True,
            use_timestep_embedding=False,
            time_embedding_type="fourier",
            mid_block_type="UNetMidBlock1D",
            down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
        )
        scheduler = IPNDMScheduler()

        components = {
            "unet": unet,
            "scheduler": scheduler,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "batch_size": 1,
            "generator": generator,
            "num_inference_steps": 4,
        }
        return inputs

    def test_dance_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        pipe = DanceDiffusionPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        audio = output.audios

        audio_slice = audio[0, -3:, -3:]

        assert audio.shape == (1, 2, components["unet"].sample_size)
        expected_slice = np.array([-0.7265, 1.0000, -0.8388, 0.1175, 0.9498, -1.0000])
        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local()

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent(expected_max_difference=3e-3)

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


@slow
@require_torch_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_dance_diffusion(self):
        device = torch_device

        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios

        audio_slice = audio[0, -3:, -3:]

        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.0192, -0.0231, -0.0318, -0.0059, 0.0002, -0.0020])

        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2

    def test_dance_diffusion_fp16(self):
        device = torch_device

        pipe = DanceDiffusionPipeline.from_pretrained("harmonai/maestro-150k", torch_dtype=torch.float16)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.manual_seed(0)
        output = pipe(generator=generator, num_inference_steps=100, audio_length_in_s=4.096)
        audio = output.audios

        audio_slice = audio[0, -3:, -3:]

        assert audio.shape == (1, 2, pipe.unet.sample_size)
        expected_slice = np.array([-0.0367, -0.0488, -0.0771, -0.0525, -0.0444, -0.0341])

        assert np.abs(audio_slice.flatten() - expected_slice).max() < 1e-2
