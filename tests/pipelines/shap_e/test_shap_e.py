# Copyright 2025 HuggingFace Inc.
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

from diffusers import HeunDiscreteScheduler, PriorTransformer, ShapEPipeline
from diffusers.pipelines.shap_e import ShapERenderer

from ...testing_utils import (
    backend_empty_cache,
    load_numpy,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference


class ShapEPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ShapEPipeline
    params = ["prompt"]
    batch_params = ["prompt"]
    required_optional_params = [
        "num_images_per_prompt",
        "num_inference_steps",
        "generator",
        "latents",
        "guidance_scale",
        "frame_size",
        "output_type",
        "return_dict",
    ]
    test_xformers_attention = False

    @property
    def text_embedder_hidden_size(self):
        return 16

    @property
    def time_input_dim(self):
        return 16

    @property
    def time_embed_dim(self):
        return self.time_input_dim * 4

    @property
    def renderer_dim(self):
        return 8

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
            "attention_head_dim": 16,
            "embedding_dim": self.time_input_dim,
            "num_embeddings": 32,
            "embedding_proj_dim": self.text_embedder_hidden_size,
            "time_embed_dim": self.time_embed_dim,
            "num_layers": 1,
            "clip_embed_dim": self.time_input_dim * 2,
            "additional_embeddings": 0,
            "time_embed_act_fn": "gelu",
            "norm_in_type": "layer",
            "encoder_hid_proj_type": None,
            "added_emb_type": None,
        }

        model = PriorTransformer(**model_kwargs)
        return model

    @property
    def dummy_renderer(self):
        torch.manual_seed(0)

        model_kwargs = {
            "param_shapes": (
                (self.renderer_dim, 93),
                (self.renderer_dim, 8),
                (self.renderer_dim, 8),
                (self.renderer_dim, 8),
            ),
            "d_latent": self.time_input_dim,
            "d_hidden": self.renderer_dim,
            "n_output": 12,
            "background": (
                0.1,
                0.1,
                0.1,
            ),
        }
        model = ShapERenderer(**model_kwargs)
        return model

    def get_dummy_components(self):
        prior = self.dummy_prior
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        shap_e_renderer = self.dummy_renderer

        scheduler = HeunDiscreteScheduler(
            beta_schedule="exp",
            num_train_timesteps=1024,
            prediction_type="sample",
            use_karras_sigmas=True,
            clip_sample=True,
            clip_sample_range=1.0,
        )
        components = {
            "prior": prior,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "shap_e_renderer": shap_e_renderer,
            "scheduler": scheduler,
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
            "num_inference_steps": 1,
            "frame_size": 32,
            "output_type": "latent",
        }
        return inputs

    def test_shap_e(self):
        device = "cpu"

        components = self.get_dummy_components()

        pipe = self.pipeline_class(**components)
        pipe = pipe.to(device)

        pipe.set_progress_bar_config(disable=None)

        output = pipe(**self.get_dummy_inputs(device))
        image = output.images[0]
        image = image.cpu().numpy()
        image_slice = image[-3:, -3:]

        assert image.shape == (32, 16)

        expected_slice = np.array([-1.0000, -0.6559, 1.0000, -0.9096, -0.7252, 0.8211, -0.7647, -0.3308, 0.6462])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_batch_consistent(self):
        # NOTE: Larger batch sizes cause this test to timeout, only test on smaller batches
        self._test_inference_batch_consistent(batch_sizes=[1, 2])

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(batch_size=2, expected_max_diff=6e-3)

    def test_num_images_per_prompt(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_size = 1
        num_images_per_prompt = 2

        inputs = self.get_dummy_inputs(torch_device)

        for key in inputs.keys():
            if key in self.batch_params:
                inputs[key] = batch_size * [inputs[key]]

        images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

        assert images.shape[0] == batch_size * num_images_per_prompt

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=5e-3)

    @unittest.skip("Key error is raised with accelerate")
    def test_sequential_cpu_offload_forward_pass(self):
        pass


@nightly
@require_torch_accelerator
class ShapEPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_shap_e(self):
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/shap_e/test_shap_e_np_out.npy"
        )
        pipe = ShapEPipeline.from_pretrained("openai/shap-e")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)

        images = pipe(
            "a shark",
            generator=generator,
            guidance_scale=15.0,
            num_inference_steps=64,
            frame_size=64,
            output_type="np",
        ).images[0]

        assert images.shape == (20, 64, 64, 3)

        assert_mean_pixel_difference(images, expected_image)
