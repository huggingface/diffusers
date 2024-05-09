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

from diffusers import DDPMWuerstchenScheduler, StableCascadeCombinedPipeline
from diffusers.models import StableCascadeUNet
from diffusers.pipelines.wuerstchen import PaellaVQModel
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class StableCascadeCombinedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableCascadeCombinedPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "prior_guidance_scale",
        "decoder_guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "prior_num_inference_steps",
        "output_type",
    ]
    test_xformers_attention = True

    @property
    def text_embedder_hidden_size(self):
        return 32

    @property
    def dummy_prior(self):
        torch.manual_seed(0)

        model_kwargs = {
            "conditioning_dim": 128,
            "block_out_channels": (128, 128),
            "num_attention_heads": (2, 2),
            "down_num_layers_per_block": (1, 1),
            "up_num_layers_per_block": (1, 1),
            "clip_image_in_channels": 768,
            "switch_level": (False,),
            "clip_text_in_channels": self.text_embedder_hidden_size,
            "clip_text_pooled_in_channels": self.text_embedder_hidden_size,
        }

        model = StableCascadeUNet(**model_kwargs)
        return model.eval()

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
            "block_out_channels": (16, 32, 64, 128),
            "num_attention_heads": (-1, -1, 1, 2),
            "down_num_layers_per_block": (1, 1, 1, 1),
            "up_num_layers_per_block": (1, 1, 1, 1),
            "down_blocks_repeat_mappers": (1, 1, 1, 1),
            "up_blocks_repeat_mappers": (3, 3, 2, 2),
            "block_types_per_layer": (
                ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ),
            "switch_level": None,
            "clip_text_pooled_in_channels": 32,
            "dropout": (0.1, 0.1, 0.1, 0.1),
        }

        model = StableCascadeUNet(**model_kwargs)
        return model.eval()

    def get_dummy_components(self):
        prior = self.dummy_prior

        scheduler = DDPMWuerstchenScheduler()
        tokenizer = self.dummy_tokenizer
        text_encoder = self.dummy_text_encoder
        decoder = self.dummy_decoder
        vqgan = self.dummy_vqgan
        prior_text_encoder = self.dummy_text_encoder
        prior_tokenizer = self.dummy_tokenizer

        components = {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "decoder": decoder,
            "scheduler": scheduler,
            "vqgan": vqgan,
            "prior_text_encoder": prior_text_encoder,
            "prior_tokenizer": prior_tokenizer,
            "prior_prior": prior,
            "prior_scheduler": scheduler,
            "prior_feature_extractor": None,
            "prior_image_encoder": None,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "horse",
            "generator": generator,
            "prior_guidance_scale": 4.0,
            "decoder_guidance_scale": 4.0,
            "num_inference_steps": 2,
            "prior_num_inference_steps": 2,
            "output_type": "np",
            "height": 128,
            "width": 128,
        }
        return inputs

    def test_stable_cascade(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(**self.get_dummy_inputs(device), return_dict=False)[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[-3:, -3:, -1]

        assert image.shape == (1, 128, 128, 3)

        expected_slice = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_torch_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=2e-2)

    @unittest.skip(reason="fp16 not supported")
    def test_float16_inference(self):
        super().test_float16_inference()

    @unittest.skip(reason="no callback test for combined pipeline")
    def test_callback_inputs(self):
        super().test_callback_inputs()

    def test_stable_cascade_combined_prompt_embeds(self):
        device = "cpu"
        components = self.get_dummy_components()

        pipe = StableCascadeCombinedPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A photograph of a shiba inu, wearing a hat"
        (
            prompt_embeds,
            prompt_embeds_pooled,
            negative_prompt_embeds,
            negative_prompt_embeds_pooled,
        ) = pipe.prior_pipe.encode_prompt(device, 1, 1, False, prompt=prompt)
        generator = torch.Generator(device=device)

        output_prompt = pipe(
            prompt=prompt,
            num_inference_steps=1,
            prior_num_inference_steps=1,
            output_type="np",
            generator=generator.manual_seed(0),
        )
        output_prompt_embeds = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_pooled=prompt_embeds_pooled,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_pooled=negative_prompt_embeds_pooled,
            num_inference_steps=1,
            prior_num_inference_steps=1,
            output_type="np",
            generator=generator.manual_seed(0),
        )

        assert np.abs(output_prompt.images - output_prompt_embeds.images).max() < 1e-5
