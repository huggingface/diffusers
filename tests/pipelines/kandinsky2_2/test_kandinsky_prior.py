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

import inspect
import unittest

import numpy as np
import torch
from torch import nn
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import KandinskyV22PriorPipeline, PriorTransformer, UnCLIPScheduler
from diffusers.utils.testing_utils import enable_full_determinism, skip_mps, torch_device

from ..test_pipelines_common import PipelineTesterMixin


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
        inputs = {
            "prompt": "horse",
            "generator": generator,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs


class KandinskyV22PriorPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22PriorPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "generator",
        "num_inference_steps",
        "latents",
        "negative_prompt",
        "guidance_scale",
        "output_type",
        "return_dict",
    ]
    callback_cfg_params = ["prompt_embeds", "text_encoder_hidden_states", "text_mask"]
    test_xformers_attention = False

    supports_dduf = False

    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, device, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(device=device, seed=seed)

    def test_kandinsky_prior(self):
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
            [-0.5948, 0.1875, -0.1523, -1.1995, -1.4061, -0.6367, -1.4607, -0.6406, 0.8793, -0.3891]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=1e-3)

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        test_max_difference = torch_device == "cpu"
        test_mean_pixel_difference = False

        self._test_attention_slicing_forward_pass(
            test_max_difference=test_max_difference,
            test_mean_pixel_difference=test_mean_pixel_difference,
        )

    # override default test because no output_type "latent", use "pt" instead
    def test_callback_inputs(self):
        sig = inspect.signature(self.pipeline_class.__call__)

        if not ("callback_on_step_end_tensor_inputs" in sig.parameters and "callback_on_step_end" in sig.parameters):
            return

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        self.assertTrue(
            hasattr(pipe, "_callback_tensor_inputs"),
            f" {self.pipeline_class} should have `_callback_tensor_inputs` that defines a list of tensor variables its callback function can use as inputs",
        )

        def callback_inputs_test(pipe, i, t, callback_kwargs):
            missing_callback_inputs = set()
            for v in pipe._callback_tensor_inputs:
                if v not in callback_kwargs:
                    missing_callback_inputs.add(v)
            self.assertTrue(
                len(missing_callback_inputs) == 0, f"Missing callback tensor inputs: {missing_callback_inputs}"
            )
            last_i = pipe.num_timesteps - 1
            if i == last_i:
                callback_kwargs["latents"] = torch.zeros_like(callback_kwargs["latents"])
            return callback_kwargs

        inputs = self.get_dummy_inputs(torch_device)
        inputs["callback_on_step_end"] = callback_inputs_test
        inputs["callback_on_step_end_tensor_inputs"] = pipe._callback_tensor_inputs
        inputs["num_inference_steps"] = 2
        inputs["output_type"] = "pt"

        output = pipe(**inputs)[0]
        assert output.abs().sum() == 0
