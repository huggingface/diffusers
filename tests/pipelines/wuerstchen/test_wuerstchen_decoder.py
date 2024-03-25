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
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import DDPMWuerstchenScheduler, WuerstchenDecoderPipeline
from diffusers.pipelines.wuerstchen import PaellaVQModel, WuerstchenDiffNeXt
from diffusers.utils.testing_utils import enable_full_determinism, skip_mps, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class WuerstchenDecoderPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = WuerstchenDecoderPipeline
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
        return CLIPTextModel(config).eval()

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
            "c_cond": self.text_embedder_hidden_size,
            "c_hidden": [320],
            "nhead": [-1],
            "blocks": [4],
            "level_config": ["CT"],
            "clip_embd": self.text_embedder_hidden_size,
            "inject_effnet": [False],
        }

        model = WuerstchenDiffNeXt(**model_kwargs)
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
            "guidance_scale": 1.0,
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

        expected_slice = np.array([0.0000, 0.0000, 0.0089, 1.0000, 1.0000, 0.3927, 1.0000, 1.0000, 1.0000])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-5)

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        test_max_difference = torch_device == "cpu"
        test_mean_pixel_difference = False

        self._test_attention_slicing_forward_pass(
            test_max_difference=test_max_difference,
            test_mean_pixel_difference=test_mean_pixel_difference,
        )

    @unittest.skip(reason="bf16 not supported and requires CUDA")
    def test_float16_inference(self):
        super().test_float16_inference()
