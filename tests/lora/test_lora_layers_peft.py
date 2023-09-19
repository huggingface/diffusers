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
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
)
from diffusers.utils.testing_utils import floats_tensor


def create_unet_lora_layers(unet: nn.Module):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        lora_attn_processor_class = (
            LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
        )
        lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
        )
    unet_lora_layers = AttnProcsLayers(lora_attn_procs)
    return lora_attn_procs, unet_lora_layers


class LoraLoaderMixinTests(unittest.TestCase):
    def get_dummy_components(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        text_encoder_config = CLIPTextConfig(
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
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        unet_lora_attn_procs, unet_lora_layers = create_unet_lora_layers(unet)

        pipeline_components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        lora_components = {
            "unet_lora_layers": unet_lora_layers,
            "unet_lora_attn_procs": unet_lora_attn_procs,
        }
        return pipeline_components, lora_components

    def get_dummy_inputs(self, with_generator=True):
        batch_size = 1
        sequence_length = 10
        num_channels = 4
        sizes = (32, 32)

        generator = torch.manual_seed(0)
        noise = floats_tensor((batch_size, num_channels) + sizes)
        input_ids = torch.randint(1, sequence_length, size=(batch_size, sequence_length), generator=generator)

        pipeline_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        if with_generator:
            pipeline_inputs.update({"generator": generator})

        return noise, input_ids, pipeline_inputs

    # copied from: https://colab.research.google.com/gist/sayakpaul/df2ef6e1ae6d8c10a49d859883b10860/scratchpad.ipynb
    def get_dummy_tokens(self):
        max_seq_length = 77

        inputs = torch.randint(2, 56, size=(1, max_seq_length), generator=torch.manual_seed(0))

        prepared_inputs = {}
        prepared_inputs["input_ids"] = inputs
        return prepared_inputs
