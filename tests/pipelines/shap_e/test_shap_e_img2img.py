# Copyright 2026 HuggingFace Inc.
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

import numpy as np
import pytest
import torch
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModel

from diffusers import HeunDiscreteScheduler, PriorTransformer, ShapEImg2ImgPipeline
from diffusers.pipelines.shap_e import ShapERenderer

from ...testing_utils import (
    backend_empty_cache,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    MemoryTesterMixin,
    PipelineTesterMixin,
    assert_mean_pixel_difference,
)


class ShapEImg2ImgPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = ShapEImg2ImgPipeline
    required_input_params_in_call_signature = frozenset(["image"])
    batch_input_params = frozenset(["image"])
    # Shap-E renders a 3D asset, so `guidance_scale` and `frame_size` join the canonical optional parameters.
    optional_input_params = frozenset(
        [
            "num_inference_steps",
            "num_images_per_prompt",
            "generator",
            "latents",
            "guidance_scale",
            "frame_size",
            "output_type",
            "return_dict",
        ]
    )
    # The dummy inputs request latents, so a sample is the flattened latent grid rather than an image.
    output_shape = (32, 16)

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
    def dummy_image_encoder(self):
        torch.manual_seed(0)
        config = CLIPVisionConfig(
            hidden_size=self.text_embedder_hidden_size,
            image_size=32,
            projection_dim=self.text_embedder_hidden_size,
            intermediate_size=24,
            num_attention_heads=2,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=1,
        )

        model = CLIPVisionModel(config)
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
            "embedding_proj_norm_type": "layer",
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
        image_encoder = self.dummy_image_encoder
        image_processor = self.dummy_image_processor
        shap_e_renderer = self.dummy_renderer

        scheduler = HeunDiscreteScheduler(
            beta_schedule="exp",
            num_train_timesteps=1024,
            prediction_type="sample",
            use_karras_sigmas=True,
            clip_sample=True,
            clip_sample_range=1.0,
        )
        return {
            "prior": prior,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "shap_e_renderer": shap_e_renderer,
            "scheduler": scheduler,
        }

    def get_dummy_inputs(self):
        input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(0)).to(torch_device)
        return {
            "image": input_image,
            "generator": self.get_generator(0),
            "num_inference_steps": 1,
            "frame_size": 32,
            "output_type": "latent",
        }


class TestShapEImg2ImgPipeline(ShapEImg2ImgPipelineTesterConfig, PipelineTesterMixin):
    def test_shap_e(self):
        pipe = self.get_pipeline().to(torch_device)

        output = pipe(**self.get_dummy_inputs())
        image = output.images[0]
        image_slice = image[-3:, -3:].cpu().numpy()

        assert image.shape == self.output_shape

        expected_slice = np.array(
            [-1.0, -0.08255208, 1.0, -0.99469435, -0.19953543, 0.99999994, -0.9133135, 0.24063873, 0.8238962]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=2, expected_max_diff=6e-3)

    def test_num_images_per_prompt(self):
        # The base test sweeps batch size x images-per-prompt; rendering makes that too slow here, so check the
        # single interesting combination.
        pipe = self.get_pipeline().to(torch_device)

        batch_size = 1
        num_images_per_prompt = 2

        inputs = self.get_dummy_inputs()
        for key in inputs.keys():
            if key in self.batch_input_params:
                inputs[key] = batch_size * [inputs[key]]

        images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

        assert images.shape[0] == batch_size * num_images_per_prompt

    def test_save_load_local(self, tmp_path, base_pipe_output):
        super().test_save_load_local(tmp_path, base_pipe_output, expected_max_difference=5e-3)


class TestShapEImg2ImgPipelineMemory(ShapEImg2ImgPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the Shap-E img2img pipeline."""

    @pytest.mark.skip("Key error is raised with accelerate")
    def test_sequential_cpu_offload_forward_pass(self):
        pass


@nightly
@require_torch_accelerator
class TestShapEImg2ImgPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        # clean up the VRAM before and after each test
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_shap_e_img2img(self):
        input_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/shap_e/corgi.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/shap_e/test_shap_e_img2img_out.npy"
        )
        pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img")
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)

        images = pipe(
            input_image,
            generator=generator,
            guidance_scale=3.0,
            num_inference_steps=64,
            frame_size=64,
            output_type="np",
        ).images[0]

        assert images.shape == (20, 64, 64, 3)

        assert_mean_pixel_difference(images, expected_image)
