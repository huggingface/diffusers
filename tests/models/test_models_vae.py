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

import torch

from diffusers import AutoencoderKL
from diffusers.models import ModelMixin
from diffusers.utils import floats_tensor, load_hf_numpy, require_torch_gpu, slow, torch_all_close, torch_device
from parameterized import parameterized

from ..test_modeling_common import ModelTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class AutoencoderKLTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)

        return {"sample": image}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 4,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    def test_from_pretrained_hub(self):
        model, loading_info = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        model.to(torch_device)
        image = model(**self.dummy_input)

        assert image is not None, "Make sure output is not None"

    def test_output_pretrained(self):
        model = AutoencoderKL.from_pretrained("fusing/autoencoder-kl-dummy")
        model = model.to(torch_device)
        model.eval()

        # One-time warmup pass (see #372)
        if torch_device == "mps" and isinstance(model, ModelMixin):
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            image = image.to(torch_device)
            with torch.no_grad():
                _ = model(image, sample_posterior=True).sample
            generator = torch.manual_seed(0)
        else:
            generator = torch.Generator(device=torch_device).manual_seed(0)

        image = torch.randn(
            1,
            model.config.in_channels,
            model.config.sample_size,
            model.config.sample_size,
            generator=torch.manual_seed(0),
        )
        image = image.to(torch_device)
        with torch.no_grad():
            output = model(image, sample_posterior=True, generator=generator).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()

        # Since the VAE Gaussian prior's generator is seeded on the appropriate device,
        # the expected output slices are not the same for CPU and GPU.
        if torch_device == "mps":
            expected_output_slice = torch.tensor(
                [
                    -4.0078e-01,
                    -3.8323e-04,
                    -1.2681e-01,
                    -1.1462e-01,
                    2.0095e-01,
                    1.0893e-01,
                    -8.8247e-02,
                    -3.0361e-01,
                    -9.8644e-03,
                ]
            )
        elif torch_device == "cpu":
            expected_output_slice = torch.tensor(
                [-0.1352, 0.0878, 0.0419, -0.0818, -0.1069, 0.0688, -0.1458, -0.4446, -0.0026]
            )
        else:
            expected_output_slice = torch.tensor(
                [-0.2421, 0.4642, 0.2507, -0.0438, 0.0682, 0.3160, -0.2018, -0.0727, 0.2485]
            )

        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-2))


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="CompVis/stable-diffusion-v1-4", fp16=False):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.to(torch_device).eval()

        return model

    def get_generator(self, seed=0):
        if torch_device == "mps":
            return torch.Generator().manual_seed(seed)
        return torch.Generator(device=torch_device).manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824], [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824]],
            [47, [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089], [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131]],
            # fmt: on
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -0.0871, 0.5103, -0.0999]],
            [47, [-0.4128, -0.1320, -0.3704, 0.1965, -0.4116, -0.2332, -0.3340, 0.2247]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_stable_diffusion_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        image = self.get_sd_image(seed, fp16=True)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-2)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814], [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824]],
            [47, [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085], [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131]],
            # fmt: on
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)

        with torch.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.4990, -0.3720, -0.4925]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -0.1930, -0.1465, -0.2039]],
            [16, [-0.1628, -0.2134, -0.2747, -0.2642, -0.3774, -0.4404, -0.3687, -0.4277]],
            # fmt: on
        ]
    )
    @require_torch_gpu
    def test_stable_diffusion_decode_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
            # fmt: on
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)

        assert list(sample.shape) == [image.shape[0], 4] + [i // 8 for i in image.shape[2:]]

        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        tolerance = 1e-3 if torch_device != "mps" else 1e-2
        assert torch_all_close(output_slice, expected_output_slice, atol=tolerance)
