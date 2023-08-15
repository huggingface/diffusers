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
import unittest

import numpy as np
import torch

from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, DPMSolverMultistepScheduler, Transformer2DModel
from diffusers.utils import is_xformers_available, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import enable_full_determinism, require_torch_gpu

from ..pipeline_params import (
    CLASS_CONDITIONED_IMAGE_GENERATION_BATCH_PARAMS,
    CLASS_CONDITIONED_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class FabricPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = FabricDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

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
        scheduler = EulerAncestralDiscreteScheduler()
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
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

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "random_ssed": generator,
            "num_images": 1,
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        sd_pipe = FabricPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5756, 0.6118, 0.5005, 0.5041, 0.5471, 0.4726, 0.4976, 0.4865, 0.4864])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


    def test_stable_diffusion_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = FabricPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs(torch_device)
        prompt = 3 * [inputs.pop("prompt")]

        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = sd_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=sd_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = text_inputs["input_ids"].to(torch_device)

            embeds.append(sd_pipe.text_encoder(text_inputs)[0])

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4


@require_torch_gpu
@slow
class FABRICPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_fabric(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")
        pipe.to("cuda")

        prompt = "a photograph of an astronaut riding a horse"

        images = pipe(prompt, random_seed=generator).images

        for word, image in zip(prompt, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/fabric_wo_feedback.npy"
            )
            assert np.abs((expected_image - image).max()) < 1e-2

    def test_fabric_feedback(self):
        generator = torch.manual_seed(0)

        pipe = DiTPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")
        pipe.to("cuda")

        prompt = "a photograph of an astronaut riding a horse"
        images = pipe(prompt, random_seed=generator).images

        liked = [images]
        images = pipe(prompt, random_seed=generator, liked=liked).images

        for word, image in zip(prompt, images):
            expected_image = load_numpy(
                f"https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/dit/fabric_w_feedback.npy"
            )
            assert np.abs((expected_image - image).max()) < 1e-2
