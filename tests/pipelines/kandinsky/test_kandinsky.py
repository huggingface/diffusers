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

import random
import unittest

import numpy as np
import torch
from transformers import PretrainedConfig, XLMRobertaTokenizerFast

from diffusers import KandinskyPipeline, UnCLIPScheduler, UNet2DConditionModel, VQModel
from diffusers.pipelines.kandinsky.text_encoder import MultilingualCLIP
from diffusers.pipelines.kandinsky.text_proj import KandinskyTextProjModel
from diffusers.utils import floats_tensor

from ..test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)


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

    @property
    def text_embedder_hidden_size(self):
        return 1024

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
        return 100

    # YiYi's TO-DO: add a tiny tokenizer?
    @property
    def dummy_tokenizer(self):
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("YiYiXu/Kandinsky", subfolder="tokenizer")
        return tokenizer

    # @property
    # def dummy_text_encoder(self):
    #     torch.manual_seed(0)
    #     config = PretrainedConfig(
    #         modelBase="YiYiXu/tiny-random-mclip-base",
    #         numDims=100,
    #         transformerDimensions=32)

    #     return MultilingualCLIP(config)

    @property
    def dummy_text_encoder(self):
        torch.manual_seed(0)
        config = PretrainedConfig(
            modelBase="xlm-roberta-large", numDims=self.cross_attention_dim, transformerDimensions=1024
        )

        return MultilingualCLIP(config)

    @property
    def dummy_text_proj(self):
        torch.manual_seed(0)

        model_kwargs = {
            "clip_embeddings_dim": self.cross_attention_dim,
            "time_embed_dim": self.time_embed_dim,
            "clip_extra_context_tokens": 2,
            "cross_attention_dim": self.cross_attention_dim,
            "clip_text_encoder_hidden_states_dim": self.text_embedder_hidden_size,
        }

        model = KandinskyTextProjModel(**model_kwargs)
        return model

    @property
    def dummy_unet(self):
        torch.manual_seed(0)

        model_kwargs = {
            "in_channels": 4,
            # Out channels is double in channels because predicts mean and variance
            "out_channels": 8,
            "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
            "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
            "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
            "block_out_channels": (self.block_out_channels_0, self.block_out_channels_0 * 2),
            "layers_per_block": 1,
            "cross_attention_dim": self.cross_attention_dim,
            "attention_head_dim": 4,
            "resnet_time_scale_shift": "scale_shift",
            "class_embed_type": "identity",
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
        text_proj = self.dummy_text_proj
        movq = self.dummy_movq

        scheduler = UnCLIPScheduler(
            clip_sample=True,
            clip_sample_range=2.0,
            sample_min_value=1.0,
            sample_max_value=None,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            variance_type="learned_range",
            thresholding=True,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
        )

        components = {
            "text_proj": text_proj,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "movq": movq,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        image_embeds = floats_tensor((1, self.cross_attention_dim), rng=random.Random(seed)).to(device)
        floats_tensor((1, self.cross_attention_dim), rng=random.Random(seed + 1)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "horse",
            "image_embeds": image_embeds,
            "negative_image_embeds": image_embeds,
            "generator": generator,
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

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

        print(f"image.shape {image.shape}")

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array(
            [0.5208529, 0.4821977, 0.44796965, 0.5479469, 0.54242486, 0.45028442, 0.42460358, 0.46456948, 0.48675597]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2
