# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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
from parameterized import parameterized
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    LCMScheduler,
    MultiAdapter,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import PipelineFromPipeTesterMixin, PipelineTesterMixin, assert_mean_pixel_difference


enable_full_determinism()


class AdapterTests:
    pipeline_class = StableDiffusionAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self, adapter_type, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            time_cond_proj_dim=time_cond_proj_dim,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
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

        torch.manual_seed(0)

        if adapter_type == "full_adapter" or adapter_type == "light_adapter":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 64],
                num_res_blocks=2,
                downscale_factor=2,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=2,
                        adapter_type="full_adapter",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=2,
                        adapter_type="full_adapter",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter', 'light_adapter', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_components_with_full_downscaling(self, adapter_type):
        """Get dummy components with x8 VAE downscaling and 4 UNet down blocks.
        These dummy components are intended to fully-exercise the T2I-Adapter
        downscaling behavior.
        """
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
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

        torch.manual_seed(0)

        if adapter_type == "full_adapter" or adapter_type == "light_adapter":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 32, 32, 64],
                num_res_blocks=2,
                downscale_factor=8,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=8,
                        adapter_type="full_adapter",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=8,
                        adapter_type="full_adapter",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter', 'light_adapter', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0, height=64, width=64, num_images=1):
        if num_images == 1:
            image = floats_tensor((1, 3, height, width), rng=random.Random(seed)).to(device)
        else:
            image = [
                floats_tensor((1, 3, height, width), rng=random.Random(seed)).to(device) for _ in range(num_images)
            ]

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=2e-3)

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=2e-3)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=2e-3)

    @parameterized.expand(
        [
            # (dim=264) The internal feature map will be 33x33 after initial pixel unshuffling (downscaled x8).
            (((4 * 8 + 1) * 8),),
            # (dim=272) The internal feature map will be 17x17 after the first T2I down block (downscaled x16).
            (((4 * 4 + 1) * 16),),
            # (dim=288) The internal feature map will be 9x9 after the second T2I down block (downscaled x32).
            (((4 * 2 + 1) * 32),),
            # (dim=320) The internal feature map will be 5x5 after the third T2I down block (downscaled x64).
            (((4 * 1 + 1) * 64),),
        ]
    )
    def test_multiple_image_dimensions(self, dim):
        """Test that the T2I-Adapter pipeline supports any input dimension that
        is divisible by the adapter's `downscale_factor`. This test was added in
        response to an issue where the T2I Adapter's downscaling padding
        behavior did not match the UNet's behavior.

        Note that we have selected `dim` values to produce odd resolutions at
        each downscaling level.
        """
        components = self.get_dummy_components_with_full_downscaling()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device, height=dim, width=dim)
        image = sd_pipe(**inputs).images

        assert image.shape == (1, dim, dim, 3)

    def test_adapter_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4535, 0.5493, 0.4359, 0.5452, 0.6086, 0.4441, 0.5544, 0.501, 0.4859])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_adapter_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4535, 0.5493, 0.4359, 0.5452, 0.6086, 0.4441, 0.5544, 0.501, 0.4859])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


class StableDiffusionFullAdapterPipelineFastTests(
    AdapterTests, PipelineTesterMixin, PipelineFromPipeTesterMixin, unittest.TestCase
):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("full_adapter", time_cond_proj_dim=time_cond_proj_dim)

    def get_dummy_components_with_full_downscaling(self):
        return super().get_dummy_components_with_full_downscaling("full_adapter")

    def test_stable_diffusion_adapter_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4858, 0.5500, 0.4278, 0.4669, 0.6184, 0.4322, 0.5010, 0.5033, 0.4746])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3

    def test_from_pipe_consistent_forward_pass_cpu_offload(self):
        super().test_from_pipe_consistent_forward_pass_cpu_offload(expected_max_diff=6e-3)


class StableDiffusionLightAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("light_adapter", time_cond_proj_dim=time_cond_proj_dim)

    def get_dummy_components_with_full_downscaling(self):
        return super().get_dummy_components_with_full_downscaling("light_adapter")

    def test_stable_diffusion_adapter_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4965, 0.5548, 0.4330, 0.4771, 0.6226, 0.4382, 0.5037, 0.5071, 0.4782])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3


class StableDiffusionMultiAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("multi_adapter", time_cond_proj_dim=time_cond_proj_dim)

    def get_dummy_components_with_full_downscaling(self):
        return super().get_dummy_components_with_full_downscaling("multi_adapter")

    def get_dummy_inputs(self, device, height=64, width=64, seed=0):
        inputs = super().get_dummy_inputs(device, seed, height=height, width=width, num_images=2)
        inputs["adapter_conditioning_scale"] = [0.5, 0.5]
        return inputs

    def test_stable_diffusion_adapter_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4902, 0.5539, 0.4317, 0.4682, 0.6190, 0.4351, 0.5018, 0.5046, 0.4772])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3

    def test_inference_batch_consistent(
        self, batch_sizes=[2, 4, 13], additional_params_copy_to_batched_inputs=["num_inference_steps"]
    ):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        for batch_size in batch_sizes:
            batched_inputs = {}
            for name, value in inputs.items():
                if name in self.batch_params:
                    # prompt is string
                    if name == "prompt":
                        len_prompt = len(value)
                        # make unequal batch sizes
                        batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                        # make last batch super long
                        batched_inputs[name][-1] = 100 * "very long"
                    elif name == "image":
                        batched_images = []

                        for image in value:
                            batched_images.append(batch_size * [image])

                        batched_inputs[name] = batched_images
                    else:
                        batched_inputs[name] = batch_size * [value]

                elif name == "batch_size":
                    batched_inputs[name] = batch_size
                else:
                    batched_inputs[name] = value

            for arg in additional_params_copy_to_batched_inputs:
                batched_inputs[arg] = inputs[arg]

            batched_inputs["output_type"] = "np"

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)

            assert len(output[0]) == batch_size

            batched_inputs["output_type"] = "np"

            if self.pipeline_class.__name__ == "DanceDiffusionPipeline":
                batched_inputs.pop("output_type")

            output = pipe(**batched_inputs)[0]

            assert output.shape[0] == batch_size

        logger.setLevel(level=diffusers.logging.WARNING)

    def test_num_images_per_prompt(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        batch_sizes = [1, 2]
        num_images_per_prompts = [1, 2]

        for batch_size in batch_sizes:
            for num_images_per_prompt in num_images_per_prompts:
                inputs = self.get_dummy_inputs(torch_device)

                for key in inputs.keys():
                    if key in self.batch_params:
                        if key == "image":
                            batched_images = []

                            for image in inputs[key]:
                                batched_images.append(batch_size * [image])

                            inputs[key] = batched_images
                        else:
                            inputs[key] = batch_size * [inputs[key]]

                images = pipe(**inputs, num_images_per_prompt=num_images_per_prompt)[0]

                assert images.shape[0] == batch_size * num_images_per_prompt

    def test_inference_batch_single_identical(
        self,
        batch_size=3,
        test_max_difference=None,
        test_mean_pixel_difference=None,
        relax_max_difference=False,
        expected_max_diff=2e-3,
        additional_params_copy_to_batched_inputs=["num_inference_steps"],
    ):
        if test_max_difference is None:
            # TODO(Pedro) - not sure why, but not at all reproducible at the moment it seems
            # make sure that batched and non-batched is identical
            test_max_difference = torch_device != "mps"

        if test_mean_pixel_difference is None:
            # TODO same as above
            test_mean_pixel_difference = torch_device != "mps"

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device)

        logger = logging.get_logger(pipe.__module__)
        logger.setLevel(level=diffusers.logging.FATAL)

        # batchify inputs
        batched_inputs = {}
        for name, value in inputs.items():
            if name in self.batch_params:
                # prompt is string
                if name == "prompt":
                    len_prompt = len(value)
                    # make unequal batch sizes
                    batched_inputs[name] = [value[: len_prompt // i] for i in range(1, batch_size + 1)]

                    # make last batch super long
                    batched_inputs[name][-1] = 100 * "very long"
                elif name == "image":
                    batched_images = []

                    for image in value:
                        batched_images.append(batch_size * [image])

                    batched_inputs[name] = batched_images
                else:
                    batched_inputs[name] = batch_size * [value]
            elif name == "batch_size":
                batched_inputs[name] = batch_size
            elif name == "generator":
                batched_inputs[name] = [self.get_generator(i) for i in range(batch_size)]
            else:
                batched_inputs[name] = value

        for arg in additional_params_copy_to_batched_inputs:
            batched_inputs[arg] = inputs[arg]

        if self.pipeline_class.__name__ != "DanceDiffusionPipeline":
            batched_inputs["output_type"] = "np"

        output_batch = pipe(**batched_inputs)
        assert output_batch[0].shape[0] == batch_size

        inputs["generator"] = self.get_generator(0)

        output = pipe(**inputs)

        logger.setLevel(level=diffusers.logging.WARNING)
        if test_max_difference:
            if relax_max_difference:
                # Taking the median of the largest <n> differences
                # is resilient to outliers
                diff = np.abs(output_batch[0][0] - output[0][0])
                diff = diff.flatten()
                diff.sort()
                max_diff = np.median(diff[-5:])
            else:
                max_diff = np.abs(output_batch[0][0] - output[0][0]).max()
            assert max_diff < expected_max_diff

        if test_mean_pixel_difference:
            assert_mean_pixel_difference(output_batch[0][0], output[0][0])


@slow
@require_torch_gpu
class StableDiffusionAdapterPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_stable_diffusion_adapter_depth_sd_v15(self):
        adapter_model = "TencentARC/t2iadapter_depth_sd15v2"
        sd_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        prompt = "desk"
        image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/desk_depth.png"
        input_channels = 3
        out_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_depth_sd15v2.npy"
        out_url = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_adapter/sd_adapter_v15_zoe_depth.npy"

        image = load_image(image_url)
        expected_out = load_numpy(out_url)
        if input_channels == 1:
            image = image.convert("L")

        adapter = T2IAdapter.from_pretrained(adapter_model, torch_dtype=torch.float16)

        pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter=adapter, safety_checker=None)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device="cpu").manual_seed(0)
        out = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=2, output_type="np").images

        max_diff = numpy_cosine_similarity_distance(out.flatten(), expected_out.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_adapter_zoedepth_sd_v15(self):
        adapter_model = "TencentARC/t2iadapter_zoedepth_sd15v1"
        sd_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        prompt = "motorcycle"
        image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/motorcycle.png"
        input_channels = 3
        out_url = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_adapter/sd_adapter_v15_zoe_depth.npy"

        image = load_image(image_url)
        expected_out = load_numpy(out_url)
        if input_channels == 1:
            image = image.convert("L")

        adapter = T2IAdapter.from_pretrained(adapter_model, torch_dtype=torch.float16)

        pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter=adapter, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device="cpu").manual_seed(0)
        out = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=2, output_type="np").images

        max_diff = numpy_cosine_similarity_distance(out.flatten(), expected_out.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_adapter_canny_sd_v15(self):
        adapter_model = "TencentARC/t2iadapter_canny_sd15v2"
        sd_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        prompt = "toy"
        image_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png"
        input_channels = 1
        out_url = "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_adapter/sd_adapter_v15_zoe_depth.npy"

        image = load_image(image_url)
        expected_out = load_numpy(out_url)
        if input_channels == 1:
            image = image.convert("L")

        adapter = T2IAdapter.from_pretrained(adapter_model, torch_dtype=torch.float16)

        pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter=adapter, safety_checker=None)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device="cpu").manual_seed(0)

        out = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=2, output_type="np").images

        max_diff = numpy_cosine_similarity_distance(out.flatten(), expected_out.flatten())
        assert max_diff < 1e-2

    def test_stable_diffusion_adapter_sketch_sd15(self):
        adapter_model = "TencentARC/t2iadapter_sketch_sd15v2"
        sd_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        prompt = "cat"
        image_url = (
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/edge.png"
        )
        input_channels = 1
        out_url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_sketch_sd15v2.npy"

        image = load_image(image_url)
        expected_out = load_numpy(out_url)
        if input_channels == 1:
            image = image.convert("L")

        adapter = T2IAdapter.from_pretrained(adapter_model, torch_dtype=torch.float16)

        pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter=adapter, safety_checker=None)

        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        generator = torch.Generator(device="cpu").manual_seed(0)

        out = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=2, output_type="np").images

        max_diff = numpy_cosine_similarity_distance(out.flatten(), expected_out.flatten())
        assert max_diff < 1e-2
