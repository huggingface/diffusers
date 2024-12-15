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

import gc
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import DDPMWuerstchenScheduler, StableCascadeDecoderPipeline
from diffusers.models import StableCascadeUNet
from diffusers.pipelines.wuerstchen import PaellaVQModel
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_numpy,
    load_pt,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import randn_tensor

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class StableCascadeDecoderPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableCascadeDecoderPipeline
    params = ["prompt"]
    batch_params = ["image_embeddings", "prompt", "negative_prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "num_inference_steps",
        "latents",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False
    callback_cfg_params = ["image_embeddings", "text_encoder_hidden_states"]

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def time_input_dim(self):
        return 32

    @property
    def block_out_channels_0(self):
        return self.time_input_dim

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

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
            projection_dim=self.text_embedder_hidden_size,
            hidden_size=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config).eval()

    @property
    def dummy_vqgan(self):
        torch.manual_seed(0)

        model_kwargs = {
            "bottleneck_blocks": 1,
            "num_vq_embeddings": 2,
        }
        model = PaellaVQModel(**model_kwargs)
        return model.eval()

    @property
    def dummy_decoder(self):
        torch.manual_seed(0)
        model_kwargs = {
            "in_channels": 4,
            "out_channels": 4,
            "conditioning_dim": 128,
            "block_out_channels": [16, 32, 64, 128],
            "num_attention_heads": [-1, -1, 1, 2],
            "down_num_layers_per_block": [1, 1, 1, 1],
            "up_num_layers_per_block": [1, 1, 1, 1],
            "down_blocks_repeat_mappers": [1, 1, 1, 1],
            "up_blocks_repeat_mappers": [3, 3, 2, 2],
            "block_types_per_layer": [
                ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
                ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
                ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
                ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
            ],
            "switch_level": None,
            "clip_text_pooled_in_channels": 32,
            "dropout": [0.1, 0.1, 0.1, 0.1],
        }
        model = StableCascadeUNet(**model_kwargs)
        return model.eval()

    def get_dummy_components(self):
        decoder = self.dummy_decoder
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        vqgan = self.dummy_vqgan

        scheduler = DDPMWuerstchenScheduler()

        components = {
            "decoder": decoder,
            "vqgan": vqgan,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "latent_dim_scale": 4.0,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "image_embeddings": torch.ones((1, 4, 4, 4), device=device),
            "prompt": "horse",
            "generator": generator,
            "guidance_scale": 2.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_wuerstchen_decoder(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(**self.get_dummy_inputs(device), return_dict=False)

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-2)

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        test_max_difference = torch_device == "cpu"
        test_mean_pixel_difference = False

        self._test_attention_slicing_forward_pass(
            test_max_difference=test_max_difference,
            test_mean_pixel_difference=test_mean_pixel_difference,
        )

    @unittest.skip(reason="fp16 not supported")
    def test_float16_inference(self):
        super().test_float16_inference()

    def test_stable_cascade_decoder_prompt_embeds(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = StableCascadeDecoderPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image_embeddings = inputs["image_embeddings"]
        prompt = "A photograph of a shiba inu, wearing a hat"
        (
            prompt_embeds,
            prompt_embeds_pooled,
            negative_prompt_embeds,
            negative_prompt_embeds_pooled,
        ) = pipe.encode_prompt(device, 1, 1, False, prompt=prompt)
        generator = torch.Generator(device=device)

        decoder_output_prompt = pipe(
            image_embeddings=image_embeddings,
            prompt=prompt,
            num_inference_steps=1,
            output_type="np",
            generator=generator.manual_seed(0),
        )
        decoder_output_prompt_embeds = pipe(
            image_embeddings=image_embeddings,
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            num_inference_steps=1,
            output_type="np",
            generator=generator.manual_seed(0),
        )

        assert np.abs(decoder_output_prompt.images - decoder_output_prompt_embeds.images).max() < 1e-5

    def test_stable_cascade_decoder_single_prompt_multiple_image_embeddings(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = StableCascadeDecoderPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        prior_num_images_per_prompt = 2
        decoder_num_images_per_prompt = 2
        prompt = ["a cat"]
        batch_size = len(prompt)

        generator = torch.Generator(device)
        image_embeddings = randn_tensor(
            (batch_size * prior_num_images_per_prompt, 4, 4, 4), generator=generator.manual_seed(0)
        )
        decoder_output = pipe(
            image_embeddings=image_embeddings,
            prompt=prompt,
            num_inference_steps=1,
            output_type="np",
            guidance_scale=0.0,
            generator=generator.manual_seed(0),
            num_images_per_prompt=decoder_num_images_per_prompt,
        )

        assert decoder_output.images.shape[0] == (
            batch_size * prior_num_images_per_prompt * decoder_num_images_per_prompt
        )

    def test_stable_cascade_decoder_single_prompt_multiple_image_embeddings_with_guidance(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = StableCascadeDecoderPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        prior_num_images_per_prompt = 2
        decoder_num_images_per_prompt = 2
        prompt = ["a cat"]
        batch_size = len(prompt)

        generator = torch.Generator(device)
        image_embeddings = randn_tensor(
            (batch_size * prior_num_images_per_prompt, 4, 4, 4), generator=generator.manual_seed(0)
        )
        decoder_output = pipe(
            image_embeddings=image_embeddings,
            prompt=prompt,
            num_inference_steps=1,
            output_type="np",
            guidance_scale=2.0,
            generator=generator.manual_seed(0),
            num_images_per_prompt=decoder_num_images_per_prompt,
        )

        assert decoder_output.images.shape[0] == (
            batch_size * prior_num_images_per_prompt * decoder_num_images_per_prompt
        )


@slow
@require_torch_gpu
class StableCascadeDecoderPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_cascade_decoder(self):
        pipe = StableCascadeDecoderPipeline.from_pretrained(
            "stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

        generator = torch.Generator(device="cpu").manual_seed(0)
        image_embedding = load_pt(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_cascade/image_embedding.pt"
        )

        image = pipe(
            prompt=prompt,
            image_embeddings=image_embedding,
            output_type="np",
            num_inference_steps=2,
            generator=generator,
        ).images[0]

        assert image.shape == (1024, 1024, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_cascade/stable_cascade_decoder_image.npy"
        )
        max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())
        assert max_diff < 1e-4
