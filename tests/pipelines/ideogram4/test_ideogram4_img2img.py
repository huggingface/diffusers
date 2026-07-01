# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest
from types import SimpleNamespace

import torch

from diffusers import Ideogram4Img2ImgPipeline


class DummyScheduler:
    order = 1

    def __init__(self):
        self.timesteps = torch.tensor([50, 40, 30, 20, 10])
        self.begin_index = None
        self.image_latents = None

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def scale_noise(self, image_latents, timestep, noise):
        self.image_latents = image_latents
        return image_latents


class Ideogram4Img2ImgPipelineTests(unittest.TestCase):
    def test_get_timesteps(self):
        scheduler = DummyScheduler()

        pipeline = SimpleNamespace(scheduler=scheduler)
        timesteps, num_inference_steps, t_start = Ideogram4Img2ImgPipeline.get_timesteps(
            pipeline, num_inference_steps=5, strength=0.6
        )

        self.assertEqual(num_inference_steps, 3)
        self.assertEqual(t_start, 2)
        self.assertEqual(scheduler.begin_index, 2)
        self.assertTrue(torch.equal(timesteps, torch.tensor([30, 20, 10])))

    def test_prepare_latents_packs_and_expands_image(self):
        scheduler = DummyScheduler()
        vae = SimpleNamespace(
            dtype=torch.float32,
            bn=SimpleNamespace(running_mean=torch.zeros(4), running_var=torch.ones(4)),
            config=SimpleNamespace(batch_norm_eps=0.0),
        )
        pipeline = SimpleNamespace(vae=vae, scheduler=scheduler, patch_size=2)
        pipeline._encode_vae_image = lambda image, generator: torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

        latents = Ideogram4Img2ImgPipeline.prepare_latents(
            pipeline,
            image=torch.zeros(1, 3, 4, 4),
            timestep=torch.tensor([10, 10]),
            batch_size=1,
            num_images_per_prompt=2,
            num_image_tokens=4,
            latent_dim=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=None,
        )

        expected = torch.tensor(
            [[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]], dtype=torch.float32
        ).repeat_interleave(2, dim=0)
        self.assertTrue(torch.equal(latents, expected))
        self.assertTrue(torch.equal(scheduler.image_latents, expected))

    def test_prepare_latents_rejects_invalid_image_batch(self):
        scheduler = DummyScheduler()
        vae = SimpleNamespace(
            dtype=torch.float32,
            bn=SimpleNamespace(running_mean=torch.zeros(4), running_var=torch.ones(4)),
            config=SimpleNamespace(batch_norm_eps=0.0),
        )
        pipeline = SimpleNamespace(vae=vae, scheduler=scheduler, patch_size=2)
        pipeline._encode_vae_image = lambda image, generator: torch.zeros(image.shape[0], 1, 4, 4)

        with self.assertRaisesRegex(ValueError, "image.*batch size"):
            Ideogram4Img2ImgPipeline.prepare_latents(
                pipeline,
                image=torch.zeros(3, 3, 4, 4),
                timestep=torch.tensor([10, 10]),
                batch_size=2,
                num_images_per_prompt=1,
                num_image_tokens=4,
                latent_dim=4,
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=None,
            )
