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

import gc
import random
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineKarrasSchedulerTesterMixin, PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


class StableDiffusionLatentUpscalePipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
        "negative_prompt_embeds",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 4
        sizes = (16, 16)

        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0)).to(torch_device)
        return image

    def get_dummy_components(self):
        torch.manual_seed(0)
        model = UNet2DConditionModel(
            act_fn="gelu",
            attention_head_dim=8,
            norm_num_groups=None,
            block_out_channels=[32, 32, 64, 64],
            time_cond_proj_dim=160,
            conv_in_kernel=1,
            conv_out_kernel=1,
            cross_attention_dim=32,
            down_block_types=(
                "KDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
            ),
            in_channels=8,
            mid_block_type=None,
            only_cross_attention=False,
            out_channels=5,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 64, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=[
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        scheduler = EulerDiscreteScheduler(prediction_type="sample")
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="quick_gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        components = {
            "unet": model.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }

        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_image.cpu(),
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        device = "cpu"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        self.assertEqual(image.shape, (1, 256, 256, 3))
        expected_slice = np.array(
            [0.47222412, 0.41921633, 0.44717434, 0.46874192, 0.42588258, 0.46150726, 0.4677534, 0.45583832, 0.48579055]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 1e-3)

    def test_stable_diffusion_latent_upscaler_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionLatentUpscalePipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 256, 256, 3)
        expected_slice = np.array(
            [0.43865365, 0.404124, 0.42618454, 0.44333526, 0.40564927, 0.43818694, 0.4411913, 0.43404633, 0.46392226]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_stable_diffusion_latent_upscaler_multiple_init_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionLatentUpscalePipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * 2
        inputs["image"] = inputs["image"].repeat(2, 1, 1, 1)
        image = sd_pipe(**inputs).images
        image_slice = image[-1, -3:, -3:, -1]

        assert image.shape == (2, 256, 256, 3)
        expected_slice = np.array(
            [0.38730142, 0.35695046, 0.40646142, 0.40967226, 0.3981609, 0.4195988, 0.4248805, 0.430259, 0.45694894]
        )

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-3

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=7e-3)

    def test_sequential_cpu_offload_forward_pass(self):
        super().test_sequential_cpu_offload_forward_pass(expected_max_diff=3e-3)

    def test_dict_tuple_outputs_equivalent(self):
        super().test_dict_tuple_outputs_equivalent(expected_max_difference=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=7e-3)

    def test_pt_np_pil_outputs_equivalent(self):
        super().test_pt_np_pil_outputs_equivalent(expected_max_diff=3e-3)

    def test_save_load_local(self):
        super().test_save_load_local(expected_max_difference=3e-3)

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components(expected_max_difference=3e-3)

    def test_karras_schedulers_shape(self):
        skip_schedulers = [
            "DDIMScheduler",
            "DDPMScheduler",
            "PNDMScheduler",
            "HeunDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "DPMSolverSDEScheduler",
            "EDMEulerScheduler",
        ]
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        # make sure that PNDM does not need warm-up
        pipe.scheduler.register_to_config(skip_prk_steps=True)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 2

        outputs = []
        for scheduler_enum in KarrasDiffusionSchedulers:
            if scheduler_enum.name in skip_schedulers:
                # no sigma schedulers are not supported
                # no schedulers
                continue

            scheduler_cls = getattr(diffusers, scheduler_enum.name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            output = pipe(**inputs)[0]
            outputs.append(output)

        assert check_same_shape(outputs)

    def test_float16_inference(self):
        super().test_float16_inference(expected_max_diff=5e-1)

    @unittest.skip("Test not supported for a weird use of `text_input_ids`.")
    def test_encode_prompt_works_in_isolation(self):
        pass


@require_torch_accelerator
@slow
class StableDiffusionLatentUpscalePipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_latent_upscaler_fp16(self):
        generator = torch.manual_seed(33)

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
        pipe.to(torch_device)

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to(torch_device)

        prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"

        low_res_latents = pipe(prompt, generator=generator, output_type="latent").images

        image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/astronaut_1024.npy"
        )
        assert np.abs((expected_image - image).mean()) < 5e-2

    def test_latent_upscaler_fp16_image(self):
        generator = torch.manual_seed(33)

        upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
            "stabilityai/sd-x2-latent-upscaler", torch_dtype=torch.float16
        )
        upscaler.to(torch_device)

        prompt = "the temple of fire by Ross Tran and Gerardo Dottori, oil on canvas"

        low_res_img = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_512.png"
        )

        image = upscaler(
            prompt=prompt,
            image=low_res_img,
            num_inference_steps=20,
            guidance_scale=0,
            generator=generator,
            output_type="np",
        ).images[0]

        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/latent-upscaler/fire_temple_1024.npy"
        )
        assert np.abs((expected_image - image).max()) < 5e-2
