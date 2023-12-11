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
from parameterized import parameterized
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    LCMScheduler,
    MultiAdapter,
    StableDiffusionXLAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from diffusers.utils import load_image, logging
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    numpy_cosine_similarity_distance,
    require_torch_gpu,
    slow,
    torch_device,
)

from ..pipeline_params import TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS, TEXT_GUIDED_IMAGE_VARIATION_PARAMS
from ..test_pipelines_common import (
    PipelineTesterMixin,
    SDXLOptionalComponentsTesterMixin,
    assert_mean_pixel_difference,
)


enable_full_determinism()


class StableDiffusionXLAdapterPipelineFastTests(
    PipelineTesterMixin, SDXLOptionalComponentsTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionXLAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self, adapter_type="full_adapter_xl", time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            time_cond_proj_dim=time_cond_proj_dim,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        if adapter_type == "full_adapter_xl":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 64],
                num_res_blocks=2,
                downscale_factor=4,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=4,
                        adapter_type="full_adapter_xl",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 64],
                        num_res_blocks=2,
                        downscale_factor=4,
                        adapter_type="full_adapter_xl",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter_xl', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            # "safety_checker": None,
            # "feature_extractor": None,
        }
        return components

    def get_dummy_components_with_full_downscaling(self, adapter_type="full_adapter_xl"):
        """Get dummy components with x8 VAE downscaling and 3 UNet down blocks.
        These dummy components are intended to fully-exercise the T2I-Adapter
        downscaling behavior.
        """
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=2,
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=1,
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
        )
        torch.manual_seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            # SD2-specific config below
            hidden_act="gelu",
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        if adapter_type == "full_adapter_xl":
            adapter = T2IAdapter(
                in_channels=3,
                channels=[32, 32, 64],
                num_res_blocks=2,
                downscale_factor=16,
                adapter_type=adapter_type,
            )
        elif adapter_type == "multi_adapter":
            adapter = MultiAdapter(
                [
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=16,
                        adapter_type="full_adapter_xl",
                    ),
                    T2IAdapter(
                        in_channels=3,
                        channels=[32, 32, 64],
                        num_res_blocks=2,
                        downscale_factor=16,
                        adapter_type="full_adapter_xl",
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}, must be one of 'full_adapter_xl', or 'multi_adapter''"
            )

        components = {
            "adapter": adapter,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            # "safety_checker": None,
            # "feature_extractor": None,
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
            "guidance_scale": 5.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_adapter_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.5752919, 0.6022097, 0.4728038, 0.49861962, 0.57084894, 0.4644975, 0.5193715, 0.5133664, 0.4729858]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-3

    @parameterized.expand(
        [
            # (dim=144) The internal feature map will be 9x9 after initial pixel unshuffling (downscaled x16).
            (((4 * 2 + 1) * 16),),
            # (dim=160) The internal feature map will be 5x5 after the first T2I down block (downscaled x32).
            (((4 * 1 + 1) * 32),),
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
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(torch_device, height=dim, width=dim)
        image = sd_pipe(**inputs).images

        assert image.shape == (1, dim, dim, 3)

    @parameterized.expand(["full_adapter", "full_adapter_xl", "light_adapter"])
    def test_total_downscale_factor(self, adapter_type):
        """Test that the T2IAdapter correctly reports its total_downscale_factor."""
        batch_size = 1
        in_channels = 3
        out_channels = [320, 640, 1280, 1280]
        in_image_size = 512

        adapter = T2IAdapter(
            in_channels=in_channels,
            channels=out_channels,
            num_res_blocks=2,
            downscale_factor=8,
            adapter_type=adapter_type,
        )
        adapter.to(torch_device)

        in_image = floats_tensor((batch_size, in_channels, in_image_size, in_image_size)).to(torch_device)

        adapter_state = adapter(in_image)

        # Assume that the last element in `adapter_state` has been downsampled the most, and check
        # that it matches the `total_downscale_factor`.
        expected_out_image_size = in_image_size // adapter.total_downscale_factor
        assert adapter_state[-1].shape == (
            batch_size,
            out_channels[-1],
            expected_out_image_size,
            expected_out_image_size,
        )

    def test_save_load_optional_components(self):
        return self._test_save_load_optional_components()

    def test_adapter_sdxl_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5425, 0.5385, 0.4964, 0.5045, 0.6149, 0.4974, 0.5469, 0.5332, 0.5426])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_adapter_sdxl_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
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
        expected_slice = np.array([0.5425, 0.5385, 0.4964, 0.5045, 0.6149, 0.4974, 0.5469, 0.5332, 0.5426])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


class StableDiffusionXLMultiAdapterPipelineFastTests(
    StableDiffusionXLAdapterPipelineFastTests, PipelineTesterMixin, unittest.TestCase
):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("multi_adapter", time_cond_proj_dim=time_cond_proj_dim)

    def get_dummy_components_with_full_downscaling(self):
        return super().get_dummy_components_with_full_downscaling("multi_adapter")

    def get_dummy_inputs(self, device, seed=0, height=64, width=64):
        inputs = super().get_dummy_inputs(device, seed, height, width, num_images=2)
        inputs["adapter_conditioning_scale"] = [0.5, 0.5]
        return inputs

    def test_stable_diffusion_adapter_default_case(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.5813032, 0.60995954, 0.47563356, 0.5056669, 0.57199144, 0.4631841, 0.5176794, 0.51252556, 0.47183886]
        )
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

            output = pipe(**batched_inputs)

            assert len(output[0]) == batch_size

            batched_inputs["output_type"] = "np"

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
        batch_size = batch_size
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

    def test_adapter_sdxl_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5313, 0.5375, 0.4942, 0.5021, 0.6142, 0.4968, 0.5434, 0.5311, 0.5448])

        debug = [str(round(i, 4)) for i in image_slice.flatten().tolist()]
        print(",".join(debug))

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_adapter_sdxl_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLAdapterPipeline(**components)
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
        expected_slice = np.array([0.5313, 0.5375, 0.4942, 0.5021, 0.6142, 0.4968, 0.5434, 0.5311, 0.5448])

        debug = [str(round(i, 4)) for i in image_slice.flatten().tolist()]
        print(",".join(debug))

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2


@slow
@require_torch_gpu
class AdapterSDXLPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_canny_lora(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", torch_dtype=torch.float16).to(
            "cpu"
        )
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            adapter=adapter,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors")
        pipe.enable_sequential_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array(
            [0.50346327, 0.50708383, 0.50719553, 0.5135172, 0.5155377, 0.5066059, 0.49680984, 0.5005894, 0.48509413]
        )
        assert numpy_cosine_similarity_distance(original_image, expected_image) < 1e-4
