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
from parameterized import parameterized

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.loading_utils import load_image
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_hf_numpy,
    require_torch_accelerator,
    require_torch_accelerator_with_fp16,
    require_torch_accelerator_with_training,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_all_close,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from .test_modeling_common import ModelTesterMixin, UNetTesterMixin


enable_full_determinism()


def get_autoencoder_kl_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [32, 64]
    norm_num_groups = norm_num_groups or 32
    init_dict = {
        "block_out_channels": block_out_channels,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "latent_channels": 4,
        "norm_num_groups": norm_num_groups,
    }
    return init_dict


def get_asym_autoencoder_kl_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [32, 64]
    norm_num_groups = norm_num_groups or 32
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


def get_autoencoder_tiny_config(block_out_channels=None):
    block_out_channels = (len(block_out_channels) * [32]) if block_out_channels is not None else [32, 32]
    init_dict = {
        "in_channels": 3,
        "out_channels": 3,
        "encoder_block_out_channels": block_out_channels,
        "decoder_block_out_channels": block_out_channels,
        "num_encoder_blocks": [b // min(block_out_channels) for b in block_out_channels],
        "num_decoder_blocks": [b // min(block_out_channels) for b in reversed(block_out_channels)],
    }
    return init_dict


def get_consistency_vae_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [32, 64]
    norm_num_groups = norm_num_groups or 32
    return {
        "encoder_block_out_channels": block_out_channels,
        "encoder_in_channels": 3,
        "encoder_out_channels": 4,
        "encoder_down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "decoder_add_attention": False,
        "decoder_block_out_channels": block_out_channels,
        "decoder_down_block_types": ["ResnetDownsampleBlock2D"] * len(block_out_channels),
        "decoder_downsample_padding": 1,
        "decoder_in_channels": 7,
        "decoder_layers_per_block": 1,
        "decoder_norm_eps": 1e-05,
        "decoder_norm_num_groups": norm_num_groups,
        "encoder_norm_num_groups": norm_num_groups,
        "decoder_num_train_timesteps": 1024,
        "decoder_out_channels": 6,
        "decoder_resnet_time_scale_shift": "scale_shift",
        "decoder_time_embedding_type": "learned",
        "decoder_up_block_types": ["ResnetUpsampleBlock2D"] * len(block_out_channels),
        "scaling_factor": 1,
        "latent_channels": 4,
    }


class AutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

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
        init_dict = get_autoencoder_kl_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    @require_torch_accelerator_with_training
    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()

        labels = torch.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        for name, param in named_params.items():
            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=5e-5))

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

        # Keep generator on CPU for non-CUDA devices to compare outputs with CPU result tensors
        generator_device = "cpu" if not torch_device.startswith("cuda") else "cuda"
        if torch_device != "mps":
            generator = torch.Generator(device=generator_device).manual_seed(0)
        else:
            generator = torch.manual_seed(0)

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
        elif generator_device == "cpu":
            expected_output_slice = torch.tensor(
                [
                    -0.1352,
                    0.0878,
                    0.0419,
                    -0.0818,
                    -0.1069,
                    0.0688,
                    -0.1458,
                    -0.4446,
                    -0.0026,
                ]
            )
        else:
            expected_output_slice = torch.tensor(
                [
                    -0.2421,
                    0.4642,
                    0.2507,
                    -0.0438,
                    0.0682,
                    0.3160,
                    -0.2018,
                    -0.0727,
                    0.2485,
                ]
            )

        self.assertTrue(torch_all_close(output_slice, expected_output_slice, rtol=1e-2))


class AsymmetricAutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AsymmetricAutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

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
        init_dict = get_asym_autoencoder_kl_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_forward_with_norm_groups(self):
        pass


class AutoencoderTinyTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderTiny
    main_input_name = "sample"
    base_precision = 1e-2

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
        init_dict = get_autoencoder_tiny_config()
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_outputs_equivalence(self):
        pass


class ConsistencyDecoderVAETests(ModelTesterMixin, unittest.TestCase):
    model_class = ConsistencyDecoderVAE
    main_input_name = "sample"
    base_precision = 1e-2
    forward_requires_fresh_args = True

    def inputs_dict(self, seed=None):
        generator = torch.Generator("cpu")
        if seed is not None:
            generator.manual_seed(0)
        image = randn_tensor((4, 3, 32, 32), generator=generator, device=torch.device(torch_device))

        return {"sample": image, "generator": generator}

    @property
    def input_shape(self):
        return (3, 32, 32)

    @property
    def output_shape(self):
        return (3, 32, 32)

    @property
    def init_dict(self):
        return get_consistency_vae_config()

    def prepare_init_args_and_inputs_for_common(self):
        return self.init_dict, self.inputs_dict()

    @unittest.skip
    def test_training(self):
        ...

    @unittest.skip
    def test_ema_training(self):
        ...


class AutoncoderKLTemporalDecoderFastTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLTemporalDecoder
    main_input_name = "sample"
    base_precision = 1e-2

    @property
    def dummy_input(self):
        batch_size = 3
        num_channels = 3
        sizes = (32, 32)

        image = floats_tensor((batch_size, num_channels) + sizes).to(torch_device)
        num_frames = 3

        return {"sample": image, "num_frames": num_frames}

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
            "latent_channels": 4,
            "layers_per_block": 2,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_forward_signature(self):
        pass

    def test_training(self):
        pass

    @unittest.skipIf(torch_device == "mps", "Gradient checkpointing skipped on MPS")
    def test_gradient_checkpointing(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.to(torch_device)

        assert not model.is_gradient_checkpointing and model.training

        out = model(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model.zero_grad()

        labels = torch.randn_like(out)
        loss = (out - labels).mean()
        loss.backward()

        # re-instantiate the model now enabling gradient checkpointing
        model_2 = self.model_class(**init_dict)
        # clone model
        model_2.load_state_dict(model.state_dict())
        model_2.to(torch_device)
        model_2.enable_gradient_checkpointing()

        assert model_2.is_gradient_checkpointing and model_2.training

        out_2 = model_2(**inputs_dict).sample
        # run the backwards pass on the model. For backwards pass, for simplicity purpose,
        # we won't calculate the loss and rather backprop on out.sum()
        model_2.zero_grad()
        loss_2 = (out_2 - labels).mean()
        loss_2.backward()

        # compare the output and parameters gradients
        self.assertTrue((loss - loss_2).abs() < 1e-5)
        named_params = dict(model.named_parameters())
        named_params_2 = dict(model_2.named_parameters())
        for name, param in named_params.items():
            if "post_quant_conv" in name:
                continue

            self.assertTrue(torch_all_close(param.grad.data, named_params_2[name].grad.data, atol=5e-5))


@slow
class AutoencoderTinyIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="hf-internal-testing/taesd-diffusers", fp16=False):
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderTiny.from_pretrained(model_id, torch_dtype=torch_dtype)
        model.to(torch_device).eval()
        return model

    @parameterized.expand(
        [
            [(1, 4, 73, 97), (1, 3, 584, 776)],
            [(1, 4, 97, 73), (1, 3, 776, 584)],
            [(1, 4, 49, 65), (1, 3, 392, 520)],
            [(1, 4, 65, 49), (1, 3, 520, 392)],
            [(1, 4, 49, 49), (1, 3, 392, 392)],
        ]
    )
    def test_tae_tiling(self, in_shape, out_shape):
        model = self.get_sd_vae_model()
        model.enable_tiling()
        with torch.no_grad():
            zeros = torch.zeros(in_shape).to(torch_device)
            dec = model.decode(zeros).sample
            assert dec.shape == out_shape

    def test_stable_diffusion(self):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed=33)

        with torch.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor([0.0093, 0.6385, -0.1274, 0.1631, -0.1762, 0.5232, -0.3108, -0.0382])

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand([(True,), (False,)])
    def test_tae_roundtrip(self, enable_tiling):
        # load the autoencoder
        model = self.get_sd_vae_model()
        if enable_tiling:
            model.enable_tiling()

        # make a black image with a white square in the middle,
        # which is large enough to split across multiple tiles
        image = -torch.ones(1, 3, 1024, 1024, device=torch_device)
        image[..., 256:768, 256:768] = 1.0

        # round-trip the image through the autoencoder
        with torch.no_grad():
            sample = model(image).sample

        # the autoencoder reconstruction should match original image, sorta
        def downscale(x):
            return torch.nn.functional.avg_pool2d(x, model.spatial_scale_factor)

        assert torch_all_close(downscale(sample), downscale(image), atol=0.125)


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
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

    def get_sd_vae_model(self, model_id="CompVis/stable-diffusion-v1-4", fp16=False):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.to(torch_device)

        return model

    def get_generator(self, seed=0):
        generator_device = "cpu" if not torch_device.startswith("cuda") else "cuda"
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
            ],
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

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -0.0871, 0.5103, -0.0999]],
            [47, [-0.4128, -0.1320, -0.3704, 0.1965, -0.4116, -0.2332, -0.3340, 0.2247]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
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
            [
                33,
                [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
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
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.4990, -0.3720, -0.4925]],
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

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -0.1930, -0.1465, -0.2039]],
            [16, [-0.1628, -0.2134, -0.2747, -0.2642, -0.3774, -0.4404, -0.3687, -0.4277]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_stable_diffusion_decode_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand([(13,), (16,), (27,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )
    def test_stable_diffusion_decode_xformers_vs_2_0_fp16(self, seed):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with torch.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with torch.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert torch_all_close(sample, sample_2, atol=1e-1)

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

        assert torch_all_close(sample, sample_2, atol=1e-2)

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

    def test_stable_diffusion_model_local(self):
        model_id = "stabilityai/sd-vae-ft-mse"
        model_1 = AutoencoderKL.from_pretrained(model_id).to(torch_device)

        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        model_2 = AutoencoderKL.from_single_file(url).to(torch_device)
        image = self.get_sd_image(33)

        with torch.no_grad():
            sample_1 = model_1(image).sample
            sample_2 = model_2(image).sample

        assert sample_1.shape == sample_2.shape

        output_slice_1 = sample_1[-1, -2:, -2:, :2].flatten().float().cpu()
        output_slice_2 = sample_2[-1, -2:, -2:, :2].flatten().float().cpu()

        assert torch_all_close(output_slice_1, output_slice_2, atol=3e-3)


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
        generator_device = "cpu" if not torch_device.startswith("cuda") else "cuda"
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
            ],
            [
                47,
                [0.4400, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529],
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
            ],
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

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

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


@slow
class ConsistencyDecoderVAEIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def test_encode_decode(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        vae.to(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        ).resize((256, 256))
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[
            None, :, :, :
        ].cuda()

        latent = vae.encode(image).latent_dist.mean

        sample = vae.decode(latent, generator=torch.Generator("cpu").manual_seed(0)).sample

        actual_output = sample[0, :2, :2, :2].flatten().cpu()
        expected_output = torch.tensor([-0.0141, -0.0014, 0.0115, 0.0086, 0.1051, 0.1053, 0.1031, 0.1024])

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_sd(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", vae=vae, safety_checker=None)
        pipe.to(torch_device)

        out = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        actual_output = out[:2, :2, :2].flatten().cpu()
        expected_output = torch.tensor([0.7686, 0.8228, 0.6489, 0.7455, 0.8661, 0.8797, 0.8241, 0.8759])

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_encode_decode_f16(self):
        vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder", torch_dtype=torch.float16
        )  # TODO - update
        vae.to(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        ).resize((256, 256))
        image = (
            torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[None, :, :, :]
            .half()
            .cuda()
        )

        latent = vae.encode(image).latent_dist.mean

        sample = vae.decode(latent, generator=torch.Generator("cpu").manual_seed(0)).sample

        actual_output = sample[0, :2, :2, :2].flatten().cpu()
        expected_output = torch.tensor(
            [-0.0111, -0.0125, -0.0017, -0.0007, 0.1257, 0.1465, 0.1450, 0.1471],
            dtype=torch.float16,
        )

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_sd_f16(self):
        vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder", torch_dtype=torch.float16
        )  # TODO - update
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            vae=vae,
            safety_checker=None,
        )
        pipe.to(torch_device)

        out = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        actual_output = out[:2, :2, :2].flatten().cpu()
        expected_output = torch.tensor(
            [0.0000, 0.0249, 0.0000, 0.0000, 0.1709, 0.2773, 0.0471, 0.1035],
            dtype=torch.float16,
        )

        assert torch_all_close(actual_output, expected_output, atol=5e-3)
