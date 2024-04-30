# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import AmusedPipeline, AmusedScheduler, UVit2DModel, VQModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, slow, torch_device

from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AmusedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AmusedPipeline
    params = TEXT_TO_IMAGE_PARAMS | {"encoder_hidden_states", "negative_encoder_hidden_states"}
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS

    def get_dummy_components(self):
        torch.manual_seed(0)
        transformer = UVit2DModel(
            hidden_size=8,
            use_bias=False,
            hidden_dropout=0.0,
            cond_embed_dim=8,
            micro_cond_encode_dim=2,
            micro_cond_embed_dim=10,
            encoder_hidden_size=8,
            vocab_size=32,
            codebook_size=8,
            in_channels=8,
            block_out_channels=8,
            num_res_blocks=1,
            downsample=True,
            upsample=True,
            block_num_heads=1,
            num_hidden_layers=1,
            num_attention_heads=1,
            attention_dropout=0.0,
            intermediate_size=8,
            layer_norm_eps=1e-06,
            ln_elementwise_affine=True,
        )
        scheduler = AmusedScheduler(mask_token_id=31)
        torch.manual_seed(0)
        vqvae = VQModel(
            act_fn="silu",
            block_out_channels=[8],
            down_block_types=[
                "DownEncoderBlock2D",
            ],
            in_channels=3,
            latent_channels=8,
            layers_per_block=1,
            norm_num_groups=8,
            num_vq_embeddings=8,
            out_channels=3,
            sample_size=8,
            up_block_types=[
                "UpDecoderBlock2D",
            ],
            mid_block_add_attention=False,
            lookup_from_codebook=True,
        )
        torch.manual_seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=8,
            intermediate_size=8,
            layer_norm_eps=1e-05,
            num_attention_heads=1,
            num_hidden_layers=1,
            pad_token_id=1,
            vocab_size=1000,
            projection_dim=8,
        )
        text_encoder = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "transformer": transformer,
            "scheduler": scheduler,
            "vqvae": vqvae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
            "height": 4,
            "width": 4,
        }
        return inputs

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes, batch_generator=False)

    @unittest.skip("aMUSEd does not support lists of generators")
    def test_inference_batch_single_identical(self):
        ...


@slow
@require_torch_gpu
class AmusedPipelineSlowTests(unittest.TestCase):
    def test_amused_256(self):
        pipe = AmusedPipeline.from_pretrained("amused/amused-256")
        pipe.to(torch_device)

        image = pipe("dog", generator=torch.Generator().manual_seed(0), num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.4011, 0.3992, 0.3790, 0.3856, 0.3772, 0.3711, 0.3919, 0.3850, 0.3625])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_amused_256_fp16(self):
        pipe = AmusedPipeline.from_pretrained("amused/amused-256", variant="fp16", torch_dtype=torch.float16)
        pipe.to(torch_device)

        image = pipe("dog", generator=torch.Generator().manual_seed(0), num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0554, 0.05129, 0.0344, 0.0452, 0.0476, 0.0271, 0.0495, 0.0527, 0.0158])
        assert np.abs(image_slice - expected_slice).max() < 7e-3

    def test_amused_512(self):
        pipe = AmusedPipeline.from_pretrained("amused/amused-512")
        pipe.to(torch_device)

        image = pipe("dog", generator=torch.Generator().manual_seed(0), num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9960, 0.9960, 0.9946, 0.9980, 0.9947, 0.9932, 0.9960, 0.9961, 0.9947])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_amused_512_fp16(self):
        pipe = AmusedPipeline.from_pretrained("amused/amused-512", variant="fp16", torch_dtype=torch.float16)
        pipe.to(torch_device)

        image = pipe("dog", generator=torch.Generator().manual_seed(0), num_inference_steps=2, output_type="np").images

        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9983, 1.0, 1.0, 1.0, 1.0, 0.9989, 0.9994, 0.9976, 0.9977])
        assert np.abs(image_slice - expected_slice).max() < 3e-3
