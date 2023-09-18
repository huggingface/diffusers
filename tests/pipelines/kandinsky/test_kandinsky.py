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
import random
import unittest

import numpy as np
import torch
from transformers import XLMRobertaTokenizerFast

from diffusers import DDIMScheduler, KandinskyPipeline, KandinskyPriorPipeline, UNet2DConditionModel, VQModel
from diffusers.pipelines.kandinsky.text_encoder import MCLIPConfig, MultilingualCLIP
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_numpy,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference


enable_full_determinism()


class Dummies:
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
    def cross_attention_dim(self):
        return 32

    @property
    def dummy_tokenizer(self):
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("YiYiXu/tiny-random-mclip-base")
        return tokenizer

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = MCLIPConfig(
            numDims=self.cross_attention_dim,
            transformerDimensions=self.text_embedder_hidden_size,
            hidden_size=self.text_embedder_hidden_size,
            intermediate_size=37,
            num_attention_heads=4,
            num_hidden_layers=5,
            vocab_size=1005,
        )

        text_encoder = MultilingualCLIP(config)
        text_encoder = text_encoder.eval()

        return text_encoder

    @property
    def dummy_unet(self):
        torch.manual_seed(0)

        model_kwargs = {
            "in_channels": 4,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 8,
            "addition_embed_type": "text_image",
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "encoder_hid_dim": self.text_embedder_hidden_size,
            "encoder_hid_dim_type": "text_image_proj",
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": None,
        }

        model = UNet2DConditionModel(**model_kwargs)
        return model

    @property
    def dummy_movq_kwargs(self):
        return {
            "block_out_channels": [32, 64],
            "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 1,
            "norm_num_groups": 8,
            "norm_type": "spatial",
            "num_vq_embeddings": 12,
            "out_channels": 3,
            "up_block_types": [
                "AttnUpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }

    @property
    def dummy_movq(self):
        torch.manual_seed(0)
        model = VQModel(**self.dummy_movq_kwargs)
        return model

    def get_dummy_components(self):
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        unet = self.dummy_unet
        movq = self.dummy_movq

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
            thresholding=False,
        )

        components = {
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        image_embeds = floats_tensor((1, self.cross_attention_dim), rng=random.Random(seed)).to(device)
        negative_image_embeds = floats_tensor((1, self.cross_attention_dim), rng=random.Random(seed + 1)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "horse",
            "image_embeds": image_embeds,
            "negative_image_embeds": negative_image_embeds,
            "generator": generator,
            "height": 64,
            "width": 64,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs


class KandinskyPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyPipeline
    params = [
        "prompt",
        "image_embeds",
        "negative_image_embeds",
    ]
    batch_params = ["prompt", "negative_prompt", "image_embeds", "negative_image_embeds"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    def get_dummy_components(self):
        dummy = Dummies()
        return dummy.get_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        dummy = Dummies()
        return dummy.get_dummy_inputs(device=device, seed=seed)

    def test_kandinsky(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -3:, -3:, -1]
        image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([1.0000, 1.0000, 0.2766, 1.0000, 0.5447, 0.1737, 1.0000, 0.4316, 0.9024])

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
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3


@slow
@require_torch_gpu
class KandinskyPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_kandinsky_text2img(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/kandinsky/kandinsky_text2img_cat_fp16.npy"
        )

        pipe_prior = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
        )
        pipe_prior.to(torch_device)

        pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
        pipeline = pipeline.to(torch_device)
        pipeline.set_progress_bar_config(disable=None)

        prompt = "red cat, 4k photo"

        generator = torch.Generator(device="cuda").manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt,
            generator=generator,
            num_inference_steps=5,
            negative_prompt="",
        ).to_tuple()

        generator = torch.Generator(device="cuda").manual_seed(0)
        output = pipeline(
            prompt,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            generator=generator,
            num_inference_steps=100,
            output_type="np",
        )

        image = output.images[0]

        assert image.shape == (512, 512, 3)

        assert_mean_pixel_difference(image, expected_image)
