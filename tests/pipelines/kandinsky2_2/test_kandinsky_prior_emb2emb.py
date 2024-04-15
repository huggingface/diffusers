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

import random
import unittest

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import KandinskyV22PriorEmb2EmbPipeline, PriorTransformer, UnCLIPScheduler
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, skip_mps, torch_device

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class KandinskyV22PriorEmb2EmbPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22PriorEmb2EmbPipeline
    params = ["prompt", "image"]
    batch_params = ["prompt", "image"]
    required_optional_params = [
        "num_images_per_prompt",
        "strength",
        "generator",
        "num_inference_steps",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

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
        return 100

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
            hidden_size=self.text_embedder_hidden_size,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        return CLIPTextModelWithProjection(config)

    @property
    def dummy_prior(self):
        torch.manual_seed(0)

        model_kwargs = {
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "embedding_dim": self.text_embedder_hidden_size,
            "num_layers": 1,
        }

        model = PriorTransformer(**model_kwargs)
        # clip_std and clip_mean is initialized to be 0 so PriorTransformer.post_process_latents will always return 0 - set clip_std to be 1 so it won't return 0
        model.clip_std = nn.Parameter(torch.ones(model.clip_std.shape))
        return model

    @property
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            image_size=224,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        model = CLIPVisionModelWithProjection(config)
        return model

    @property
    def dummy_image_processor(self):
        image_processor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        return image_processor

    def get_dummy_components(self):
        prior = self.dummy_prior
        image_encoder = self.dummy_image_encoder
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        image_processor = self.dummy_image_processor

        scheduler = UnCLIPScheduler(
            variance_type="fixed_small_log",
            prediction_type="sample",
            num_train_timesteps=1000,
            clip_sample=True,
            clip_sample_range=10.0,
        )

        components = {
            "prior": prior,
            "image_encoder": image_encoder,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "scheduler": scheduler,
            "image_processor": image_processor,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))

        inputs = {
            "prompt": "horse",
            "image": init_image,
            "strength": 0.5,
            "generator": generator,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_kandinsky_prior_emb2emb(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.image_embeds

        image_from_tuple = pipe(
            **self.get_dummy_inputs(device),
            return_dict=False,
        )[0]

        image_slice = image[0, -10:]
        image_from_tuple_slice = image_from_tuple[0, -10:]

        assert image.shape == (1, 32)

        expected_slice = np.array(
            [
                0.1071284,
                1.3330271,
                0.61260223,
                -0.6691065,
                -0.3846852,
                -1.0303661,
                0.22716111,
                0.03348901,
                0.30040675,
                -0.24805029,
            ]
        )

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
