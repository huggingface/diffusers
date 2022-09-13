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

import unittest

import torch

from diffusers import AutoencoderKL, VQModel
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import floats_tensor, slow, torch_device
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .test_modeling_common import ModelTesterMixin


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


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
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
        if torch_device in ("mps", "cpu"):
            expected_output_slice = torch.tensor(
                [-0.1352, 0.0878, 0.0419, -0.0818, -0.1069, 0.0688, -0.1458, -0.4446, -0.0026]
            )
        else:
            expected_output_slice = torch.tensor(
                [-0.2421, 0.4642, 0.2507, -0.0438, 0.0682, 0.3160, -0.2018, -0.0727, 0.2485]
            )

        self.assertTrue(torch.allclose(output_slice, expected_output_slice, rtol=1e-2))


# We will verify our results on an image of cute cats
def prepare_img():
    file_path = hf_hub_download(repo_id="huggingface/cats-image", repo_type="dataset", filename="cats_image.jpeg")
    image = Image.open(file_path).convert("RGB")
    return image


@slow
class VAEIntegrationTests(unittest.TestCase):
    def test_vae_absorbing_diffusion(self):
        # TODO update to appropriate organization
        model = VQModel.from_pretrained("nielsr/absorbing-diffusion-churches", subfolder="vae")
        model.to(torch_device)

        # verify outputs on an image
        image_transformations = Compose(
            [Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        image = prepare_img()
        pixel_values = image_transformations(image).unsqueeze(0)
        pixel_values = pixel_values.to(torch_device)

        # we forward through the encoder, quantizer and decoder separately
        # since Absorbing Diffusion's VQ-GAN doesn't have quant_conv and post_quant_conv
        encoder_output = model.encoder(pixel_values)
        quant, emb_loss, info = model.quantize(encoder_output)
        decoder_output = model.decoder(quant)

        # verify encoder output
        self.assertEqual(encoder_output.shape, (1, 256, 16, 16))
        self.assertTrue(
            torch.allclose(
                encoder_output[0, 0, :3, :3],
                torch.tensor([[-1.2561, -1.1712, -1.0690], [-1.3602, -1.3631, -1.3604], [-1.3849, 0.5701, 1.2044]]),
                atol=1e-4,
            )
        )
        # verify decoder output
        self.assertEqual(decoder_output.shape, (1, 3, 256, 256))
        self.assertTrue(
            torch.allclose(
                decoder_output[0, 0, :3, :3],
                torch.tensor([[0.0985, 0.0838, 0.0922], [0.0892, 0.0787, 0.0870], [0.0840, 0.0913, 0.0964]]),
                atol=1e-4,
            )
        )
