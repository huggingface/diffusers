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

from diffusers import SpatialTransformer, VQDiffusionPipeline, VQDiffusionScheduler, VQModel
from diffusers.utils import slow
from diffusers.utils.testing_utils import require_torch_gpu
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ...test_pipelines_common import PipelineTesterMixin


class VQDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def num_embed(self):
        return 12

    @property
    def diffusion_steps(self):
        return 12

    @property
    def dummy_vqvae(self):
        torch.manual_seed(0)
        model = VQModel(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=3,
            num_vq_embeddings=self.num_embed,
        )
        return model

    @property
    def dummy_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModel(config)

    @property
    def dummy_transformer(self):
        torch.manual_seed(0)

        height = 12
        width = 12

        model = SpatialTransformer(
            n_heads=1,
            d_head=height * width,
            context_dim=32,
            discrete=True,
            num_embed=self.num_embed,
            height=height,
            width=width,
            diffusion_steps=self.diffusion_steps,
            ff_layers=["Linear", "ApproximateGELU", "Linear", "Dropout"],
            norm_layers=["AdaLayerNorm", "AdaLayerNorm", "LayerNorm"],
            attention_bias=True,
        )
        return model

    def test_vq_diffusion(self):
        device = "cpu"

        vqvae = self.dummy_vqvae
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        transformer = self.dummy_transformer
        scheduler = VQDiffusionScheduler(self.num_embed)

        pipe = VQDiffusionPipeline(
            vqvae=vqvae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "teddy bear playing in the pool"

        generator = torch.Generator(device=device).manual_seed(0)
        output = pipe([prompt], generator=generator, num_inference_steps=2, output_type="np")
        image = output.images

        generator = torch.Generator(device=device).manual_seed(0)
        image_from_tuple = pipe(
            [prompt], generator=generator, output_type="np", return_dict=False, num_inference_steps=2
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 24, 24, 3)

        expected_slice = np.array([0.6583, 0.6410, 0.5325, 0.5635, 0.5563, 0.4234, 0.6008, 0.5491, 0.4880])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_gpu
class VQDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    @unittest.skip("VQ Diffusion model not saved to hub")
    def test_vq_diffusion(self):
        pass
