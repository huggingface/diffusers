# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
from parameterized import parameterized

from diffusers import AsymmetricAutoencoderKL
from diffusers.utils.import_utils import is_xformers_available

from ...testing_utils import (
    Expectations,
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_hf_numpy,
    require_torch_accelerator,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_all_close,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class AsymmetricAutoencoderKLTests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = AsymmetricAutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

    def get_asym_autoencoder_kl_config(self, block_out_channels=None, norm_num_groups=None):
        block_out_channels = block_out_channels or [2, 4]
        norm_num_groups = norm_num_groups or 2
        init_dict = {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
            "down_block_out_channels": block_out_channels,
            "layers_per_down_block": 1,
            "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
            "up_block_out_channels": block_out_channels,
            "layers_per_up_block": 1,
            "act_fn": "silu",
            "latent_channels": 4,
            "norm_num_groups": norm_num_groups,
            "sample_size": 32,
            "scaling_factor": 0.18215,
        }
        return init_dict

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        mask = torch.ones((batch_size, 1) + sizes).to(torch_device)

        return {"sample": image, "mask": mask}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = self.get_asym_autoencoder_kl_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @unittest.skip("Unsupported test.")
    def test_forward_with_norm_groups(self):
        pass


@slow
class AsymmetricAutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="cross-attention/asymmetric-autoencoder-kl-x-1-5", fp16=False):
        revision = "main"
        torch_dtype = torch.float32

        model = AsymmetricAutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.to(torch_device).eval()

        return model

    def get_generator(self, seed=0):
        generator_device = "cpu" if not torch_device.startswith(torch_device) else torch_device
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                Expectations(
                    {
                        ("xpu", 3): torch.tensor([-0.0343, 0.2873, 0.1680, -0.0140, -0.3459, 0.3522, -0.1336, 0.1075]),
                        ("cuda", 7): torch.tensor([-0.0336, 0.3011, 0.1764, 0.0087, -0.3401, 0.3645, -0.1247, 0.1205]),
                        ("mps", None): torch.tensor(
                            [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824]
                        ),
                    }
                ),
            ],
            [
                47,
                Expectations(
                    {
                        ("xpu", 3): torch.tensor([0.4400, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529]),
                        ("cuda", 7): torch.tensor([0.4400, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529]),
                        ("mps", None): torch.tensor(
                            [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089]
                        ),
                    }
                ),
            ],
            # fmt: on
        ]
    )
    def test_stable_diffusion(self, seed, expected_slices):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()

        expected_slice = expected_slices.get_expectation()
        assert torch_all_close(output_slice, expected_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0340, 0.2870, 0.1698, -0.0105, -0.3448, 0.3529, -0.1321, 0.1097],
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
            ],
            [
                47,
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
            ],
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

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.0521, -0.2939, 0.1540, -0.1855, -0.5936, -0.3138, -0.4579, -0.2275]],
            [37, [-0.1820, -0.4345, -0.0455, -0.2923, -0.8035, -0.5089, -0.4795, -0.3106]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=2e-3)

    @parameterized.expand([(13,), (16,), (37,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )
    def test_stable_diffusion_decode_xformers_vs_2_0(self, seed):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with torch.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert torch_all_close(sample, sample_2, atol=5e-2)

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

        tolerance = 3e-3 if torch_device != "mps" else 1e-2
        assert torch_all_close(output_slice, expected_output_slice, atol=tolerance)
