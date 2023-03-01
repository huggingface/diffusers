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
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, CycleDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.utils import floats_tensor, load_image, load_numpy, slow, torch_device
from diffusers.utils.testing_utils import require_torch_gpu, skip_mps

from ...pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ...test_pipelines_common import PipelineTesterMixin


torch.backends.cuda.matmul.allow_tf32 = False


class CycleDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = CycleDiffusionPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "negative_prompt",
        "height",
        "width",
        "negative_prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

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
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "An astronaut riding an elephant",
            "source_prompt": "An astronaut riding a horse",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "eta": 0.1,
            "strength": 0.8,
            "guidance_scale": 3,
            "source_guidance_scale": 1,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_cycle(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components()
        pipe = CycleDiffusionPipeline(**components)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = pipe(**inputs)
        images = output.images

        image_slice = images[0, -3:, -3:, -1]

        assert images.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.4459, 0.4943, 0.4544, 0.6643, 0.5474, 0.4327, 0.5701, 0.5959, 0.5179])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @unittest.skipIf(torch_device != "cuda", "This test requires a GPU")
    def test_stable_diffusion_cycle_fp16(self):
        components = self.get_dummy_components()
        for name, module in components.items():
            if hasattr(module, "half"):
                components[name] = module.half()
        pipe = CycleDiffusionPipeline(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)
        output = pipe(**inputs)
        images = output.images

        image_slice = images[0, -3:, -3:, -1]

        assert images.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.3506, 0.4543, 0.446, 0.4575, 0.5195, 0.4155, 0.5273, 0.518, 0.4116])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    @skip_mps
    def test_save_load_local(self):
        return super().test_save_load_local()

    @unittest.skip("non-deterministic pipeline")
    def test_inference_batch_single_identical(self):
        return super().test_inference_batch_single_identical()

    @skip_mps
    def test_dict_tuple_outputs_equivalent(self):
        return super().test_dict_tuple_outputs_equivalent()

    @skip_mps
    def test_save_load_optional_components(self):
        return super().test_save_load_optional_components()

    @skip_mps
    def test_attention_slicing_forward_pass(self):
        return super().test_attention_slicing_forward_pass()


@slow
@require_torch_gpu
class CycleDiffusionPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_cycle_diffusion_pipeline_fp16(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/cycle-diffusion/black_colored_car.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/cycle-diffusion/blue_colored_car_fp16.npy"
        )
        init_image = init_image.resize((512, 512))

        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16, revision="fp16"
        )

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        source_prompt = "A black colored car"
        prompt = "A blue colored car"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
            generator=generator,
            output_type="np",
        )
        image = output.images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(image - expected_image).max() < 5e-1

    def test_cycle_diffusion_pipeline(self):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/cycle-diffusion/black_colored_car.png"
        )
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/cycle-diffusion/blue_colored_car.npy"
        )
        init_image = init_image.resize((512, 512))

        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = CycleDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        source_prompt = "A black colored car"
        prompt = "A blue colored car"

        generator = torch.manual_seed(0)
        output = pipe(
            prompt=prompt,
            source_prompt=source_prompt,
            image=init_image,
            num_inference_steps=100,
            eta=0.1,
            strength=0.85,
            guidance_scale=3,
            source_guidance_scale=1,
            generator=generator,
            output_type="np",
        )
        image = output.images

        assert np.abs(image - expected_image).max() < 1e-2
