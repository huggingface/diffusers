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

from diffusers import ShapEPipeline, HeunDiscreteScheduler, PriorTransformer
from diffusers.pipelines.shap_e import ShapERenderer
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu, torch_device

from transformers import CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer

from ..test_pipelines_common import PipelineTesterMixin, assert_mean_pixel_difference

enable_full_determinism()

class ShapEPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ShapEPipeline
    params = ["prompt"]
    batch_params = []
    required_optional_params = [
        "num_images_per_prompt",
        "num_inference_steps",
        "generator",
        "latents",
        "guidance_scale",
        "size",
        "ray_batch_size",
        "n_coarse_samples",
        "n_fine_samples",
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
            "upcast_softmax": True,
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
            "background": (1e-8, 1e-8, 1e-8,),
        }
        model = ShapERenderer(**model_kwargs)
        return model

    def get_dummy_components(self):
        prior = self.dummy_prior
        text_encoder = self.dummy_text_encoder
        tokenizer = self.dummy_tokenizer
        renderer = self.dummy_renderer

        scheduler = HeunDiscreteScheduler(
              beta_schedule="exp",
              num_train_timesteps=1024,
              prediction_type="sample",
              use_karras_sigmas=False,
        )
        components = {
            "prior": prior,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "renderer": renderer,
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
            "num_inference_steps": 2,
            "size":64,
            "output_type": "np",
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
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (20, 64, 64, 3)

        expected_slice = np.array(
            [
                0.00392157,
                0.00392157,
                0.00392157,
                0.00392157, 
                0.00392157, 
                0.00392157,
                0.00392157, 
                0.00392157, 
                0.00392157
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @unittest.skip(reason="Batching is not supported for this pipeline.")
    def test_inference_batch_consistent(self):
        pass

    @unittest.skip(reason="Batching is not supported for this pipeline.")
    def test_inference_batch_single_identical(self):
        pass

    # overwrite because:
    #  1. this pipeline support num_images_per_prompt but does not support batching
    #  2. this pipeline outputs 3d images, i.e a list of N lists of images, where N is our num_image_per_prompts
    def test_num_images_per_prompt(self):
        
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_size = 1
        num_images_per_prompts = [1, 2]

        
        for num_images_per_prompt in num_images_per_prompts:
            inputs = self.get_dummy_inputs(torch_device)
            images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt).images

            assert len(images) == batch_size * num_images_per_prompt