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

import numpy as np
import torch

from diffusers import ConsistencyDecoderVAE, StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    load_image,
    slow,
    torch_all_close,
    torch_device,
)
from ..test_modeling_common import ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class ConsistencyDecoderVAETests(ModelTesterMixin, AutoencoderTesterMixin, unittest.TestCase):
    model_class = ConsistencyDecoderVAE
    main_input_name = "sample"
    base_precision = 1e-2
    forward_requires_fresh_args = True

    def get_consistency_vae_config(self, block_out_channels=None, norm_num_groups=None):
        block_out_channels = block_out_channels or [2, 4]
        norm_num_groups = norm_num_groups or 2
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

    def inputs_dict(self, seed=None):
        if seed is None:
            generator = torch.Generator("cpu").manual_seed(0)
        else:
            generator = torch.Generator("cpu").manual_seed(seed)
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
        return self.get_consistency_vae_config()

    def prepare_init_args_and_inputs_for_common(self):
        return self.init_dict, self.inputs_dict()


@slow
class ConsistencyDecoderVAEIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    @torch.no_grad()
    def test_encode_decode(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        vae.to(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        ).resize((256, 256))
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[None, :, :, :].to(
            torch_device
        )

        latent = vae.encode(image).latent_dist.mean

        sample = vae.decode(latent, generator=torch.Generator("cpu").manual_seed(0)).sample

        actual_output = sample[0, :2, :2, :2].flatten().cpu()
        expected_output = torch.tensor([-0.0141, -0.0014, 0.0115, 0.0086, 0.1051, 0.1053, 0.1031, 0.1024])

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_sd(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", vae=vae, safety_checker=None
        )
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
            .to(torch_device)
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
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
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

    def test_vae_tiling(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", vae=vae, safety_checker=None, torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out_1 = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # make sure tiled vae decode yields the same result
        pipe.enable_vae_tiling()
        out_2 = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        assert torch_all_close(out_1, out_2, atol=5e-3)

        # test that tiled decode works with various shapes
        shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
        with torch.no_grad():
            for shape in shapes:
                image = torch.zeros(shape, device=torch_device, dtype=pipe.vae.dtype)
                pipe.vae.decode(image)
