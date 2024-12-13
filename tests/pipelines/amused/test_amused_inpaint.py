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

from diffusers import AmusedInpaintPipeline, AmusedScheduler, UVit2DModel, VQModel
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS, TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class AmusedInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = AmusedInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS - {"width", "height"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}

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
            codebook_size=32,
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
            down_block_types=["DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=8,
            layers_per_block=1,
            norm_num_groups=8,
            num_vq_embeddings=32,
            out_channels=3,
            sample_size=8,
            up_block_types=["UpDecoderBlock2D"],
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
        image = torch.full((1, 3, 4, 4), 1.0, dtype=torch.float32, device=device)
        mask_image = torch.full((1, 1, 4, 4), 1.0, dtype=torch.float32, device=device)
        mask_image[0, 0, 0, 0] = 0
        mask_image[0, 0, 0, 1] = 0
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
            "image": image,
            "mask_image": mask_image,
        }
        return inputs

    def test_inference_batch_consistent(self, batch_sizes=[2]):
        self._test_inference_batch_consistent(batch_sizes=batch_sizes, batch_generator=False)

    @unittest.skip("aMUSEd does not support lists of generators")
    def test_inference_batch_single_identical(self):
        ...


@slow
@require_torch_accelerator
class AmusedInpaintPipelineSlowTests(unittest.TestCase):
    def test_amused_256(self):
        pipe = AmusedInpaintPipeline.from_pretrained("amused/amused-256")
        pipe.to(torch_device)
        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((256, 256))
            .convert("RGB")
        )
        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((256, 256))
            .convert("L")
        )
        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0699, 0.0716, 0.0608, 0.0715, 0.0797, 0.0638, 0.0802, 0.0924, 0.0634])
        assert np.abs(image_slice - expected_slice).max() < 0.1

    def test_amused_256_fp16(self):
        pipe = AmusedInpaintPipeline.from_pretrained("amused/amused-256", variant="fp16", torch_dtype=torch.float16)
        pipe.to(torch_device)
        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((256, 256))
            .convert("RGB")
        )
        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((256, 256))
            .convert("L")
        )
        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.0735, 0.0749, 0.065, 0.0739, 0.0805, 0.0667, 0.0802, 0.0923, 0.0622])
        assert np.abs(image_slice - expected_slice).max() < 0.1

    def test_amused_512(self):
        pipe = AmusedInpaintPipeline.from_pretrained("amused/amused-512")
        pipe.to(torch_device)
        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((512, 512))
            .convert("RGB")
        )
        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((512, 512))
            .convert("L")
        )
        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0005, 0.0])
        assert np.abs(image_slice - expected_slice).max() < 0.05

    def test_amused_512_fp16(self):
        pipe = AmusedInpaintPipeline.from_pretrained("amused/amused-512", variant="fp16", torch_dtype=torch.float16)
        pipe.to(torch_device)
        image = (
            load_image("https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1.jpg")
            .resize((512, 512))
            .convert("RGB")
        )
        mask_image = (
            load_image(
                "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/open_muse/mountains_1_mask.png"
            )
            .resize((512, 512))
            .convert("L")
        )
        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=torch.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        ).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0227, 0.0157, 0.0098, 0.0213, 0.0250, 0.0127, 0.0280, 0.0380, 0.0095])
        assert np.abs(image_slice - expected_slice).max() < 0.003
