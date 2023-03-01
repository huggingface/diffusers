# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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

from diffusers import (
    AudioDiffusionPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    Mel,
    UNet2DModel,
)
from diffusers.utils import slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu


torch.backends.cuda.matmul.allow_tf32 = False


class PipelineFastTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def dummy_unet(self):
        torch.manual_seed(0)
        model = UNet2DModel(
            sample_size=(32, 64),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        )
        return model

    @property
    def dummy_vqvae_and_unet(self):
        torch.manual_seed(0)
        vqvae = AutoencoderKL(
            sample_size=(128, 64),
            in_channels=1,
            out_channels=1,
            latent_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        )
        unet = UNet2DModel(
            sample_size=(64, 32),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(128, 128),
            down_block_types=("AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D"),
        )
        return vqvae, unet

    def test_audio_diffusion(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        mel = Mel()

        scheduler = DDPMScheduler()
        pipe = AudioDiffusionPipeline(vqvae=None, unet=self.dummy_unet, mel=mel, scheduler=scheduler)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(generator=generator, steps=4)
        audio = output.audios[0]
        image = output.images[0]

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(generator=generator, steps=4, return_dict=False)
        image_from_tuple = output[0][0]

        assert audio.shape == (1, (self.dummy_unet.sample_size[1] - 1) * mel.hop_length)
        assert image.height == self.dummy_unet.sample_size[0] and image.width == self.dummy_unet.sample_size[1]
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        image_from_tuple_slice = np.frombuffer(image_from_tuple.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([255, 255, 255, 0, 181, 0, 124, 0, 15, 255])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() == 0

        scheduler = DDIMScheduler()
        dummy_vqvae_and_unet = self.dummy_vqvae_and_unet
        pipe = AudioDiffusionPipeline(
            vqvae=self.dummy_vqvae_and_unet[0], unet=dummy_vqvae_and_unet[1], mel=mel, scheduler=scheduler
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        np.random.seed(0)
        raw_audio = np.random.uniform(-1, 1, ((dummy_vqvae_and_unet[0].sample_size[1] - 1) * mel.hop_length,))
        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(raw_audio=raw_audio, generator=generator, start_step=5, steps=10)
        image = output.images[0]

        assert (
            image.height == self.dummy_vqvae_and_unet[0].sample_size[0]
            and image.width == self.dummy_vqvae_and_unet[0].sample_size[1]
        )
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([120, 117, 110, 109, 138, 167, 138, 148, 132, 121])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0


@slow
@require_torch_gpu
class PipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_audio_diffusion(self):
        device = torch_device

        pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-ddim-256")
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=device).manual_seed(42)
        output = pipe(generator=generator)
        audio = output.audios[0]
        image = output.images[0]

        assert audio.shape == (1, (pipe.unet.sample_size[1] - 1) * pipe.mel.hop_length)
        assert image.height == pipe.unet.sample_size[0] and image.width == pipe.unet.sample_size[1]
        image_slice = np.frombuffer(image.tobytes(), dtype="uint8")[:10]
        expected_slice = np.array([151, 167, 154, 144, 122, 134, 121, 105, 70, 26])
        assert np.abs(image_slice.flatten() - expected_slice).max() == 0
