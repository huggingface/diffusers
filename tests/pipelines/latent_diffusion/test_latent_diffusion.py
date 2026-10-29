# coding=utf-8
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

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDIMScheduler, LDMTextToImagePipeline, UNet2DConditionModel

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    load_numpy,
    nightly,
    require_torch_accelerator,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class LDMTextToImagePipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = LDMTextToImagePipeline
    # This pipeline predates prompt embeddings and negative prompting; `__call__` only takes a raw `prompt`.
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS - {
        "negative_prompt",
        "negative_prompt_embeds",
        "cross_attention_kwargs",
        "prompt_embeds",
    }
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 16, 16)
    # `__call__` has no `num_images_per_prompt`.
    optional_input_params = frozenset(["num_inference_steps", "generator", "latents", "output_type", "return_dict"])

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
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=(32, 64),
            in_channels=3,
            out_channels=3,
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
            latent_channels=4,
        )
        torch.manual_seed(0)
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

        return {
            "unet": unet,
            "scheduler": scheduler,
            "vqvae": vae,
            "bert": text_encoder,
            "tokenizer": tokenizer,
        }

    def get_dummy_inputs(self):
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }


class TestLDMTextToImagePipeline(LDMTextToImagePipelineTesterConfig, PipelineTesterMixin):
    def test_inference_text2img(self):
        # Run on CPU: the expected slice below is CPU-specific.
        pipe = self.get_pipeline()

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        # fmt: off
        expected_slice = torch.tensor([0.6511, 0.5873, 0.5183, 0.5046, 0.6663, 0.3908, 0.5729, 0.5979, 0.5058])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-3)


class TestLDMTextToImagePipelineMemory(LDMTextToImagePipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the LDM text2img pipeline."""


@nightly
@require_torch_accelerator
class TestLDMTextToImagePipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def get_inputs(self, device, dtype=torch.float32, seed=0, num_inference_steps=3):
        generator = torch.manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 32, 32))
        latents = torch.from_numpy(latents).to(device=device, dtype=dtype)
        return {
            "prompt": "A painting of a squirrel eating a burger",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

    def test_ldm_default_ddim(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256").to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_inputs(torch_device)).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array([0.51825, 0.52850, 0.52543, 0.54258, 0.52304, 0.52569, 0.54363, 0.55276, 0.56878])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_ldm_default_ddim_full_schedule(self):
        pipe = LDMTextToImagePipeline.from_pretrained("CompVis/ldm-text2im-large-256").to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image = pipe(**self.get_inputs(torch_device, num_inference_steps=50)).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/ldm_text2img/ldm_large_256_ddim.npy"
        )
        assert np.abs(expected_image - image).max() < 1e-3
