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

import copy
import random
import unittest

import numpy as np
import torch
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
)

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils.testing_utils import enable_full_determinism, floats_tensor, require_torch_gpu, slow, torch_device

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLInpaintPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionXLInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union(
        {
            "add_text_embeds",
            "add_time_ids",
            "mask",
            "masked_image_latents",
        }
    )

    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            time_cond_proj_dim=time_cond_proj_dim,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72,  # 5 * 8 + 32
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
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

        torch.manual_seed(0)
        image_encoder_config = CLIPVisionConfig(
            hidden_size=32,
            image_size=224,
            projection_dim=32,
            intermediate_size=37,
            num_attention_heads=4,
            num_channels=3,
            num_hidden_layers=5,
            patch_size=14,
        )

        image_encoder = CLIPVisionModelWithProjection(image_encoder_config)

        feature_extractor = CLIPImageProcessor(
            crop_size=224,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,
            size=224,
        )

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": image_encoder,
            "feature_extractor": feature_extractor,
            "requires_aesthetics_score": True,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        # create mask
        image[8:, 8:, :] = 255
        mask_image = Image.fromarray(np.uint8(image)).convert("L").resize((64, 64))

        if str(device).startswith("mps"):
            generator = torch.manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "strength": 1.0,
            "output_type": "np",
        }
        return inputs

    def get_dummy_inputs_2images(self, device, seed=0, img_res=64):
        # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
        image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(device)
        image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed + 22)).to(device)
        # Convert images to [-1, 1]
        init_image1 = 2.0 * image1 - 1.0
        init_image2 = 2.0 * image2 - 1.0

        # empty mask
        mask_image = torch.zeros((1, 1, img_res, img_res), device=device)

        if str(device).startswith("mps"):
            generator1 = torch.manual_seed(seed)
            generator2 = torch.manual_seed(seed)
        else:
            generator1 = torch.Generator(device=device).manual_seed(seed)
            generator2 = torch.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": ["A painting of a squirrel eating a burger"] * 2,
            "image": [init_image1, init_image2],
            "mask_image": [mask_image] * 2,
            "generator": [generator1, generator2],
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.pipeline_class(**init_components)

        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def test_stable_diffusion_xl_inpaint_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.8029, 0.5523, 0.5825, 0.6003, 0.6702, 0.7018, 0.6369, 0.5955, 0.5123])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_euler_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # TODO(Patrick, Sayak) - skip for now as this requires more refiner tests
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_inpaint_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward without prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with prompt embeds
        inputs = self.get_dummy_inputs(torch_device)
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)

        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # make sure that it's equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    @require_torch_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLInpaintPipeline(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    def test_stable_diffusion_xl_refiner(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(skip_first_text_encoder=True)

        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)

        expected_slice = np.array([0.7045, 0.4838, 0.5454, 0.6270, 0.6168, 0.6717, 0.6484, 0.5681, 0.4922])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_two_xl_mixture_of_denoiser_fast(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_ts = num_train_timesteps - int(round(num_train_timesteps * split))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts < split_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {**inputs, **{"denoising_start": split, "image": latents}}
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        for steps in [7, 20]:
            assert_run_mixture(steps, 0.33, EulerDiscreteScheduler)
            assert_run_mixture(steps, 0.33, HeunDiscreteScheduler)

    @slow
    def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_ts = num_train_timesteps - int(round(num_train_timesteps * split))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts < split_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {**inputs, **{"denoising_start": split, "image": latents}}
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        for steps in [5, 8, 20]:
            for split in [0.33, 0.49, 0.71]:
                for scheduler_cls in [
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split, scheduler_cls)

    @slow
    def test_stable_diffusion_three_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = StableDiffusionXLInpaintPipeline(**components).to(torch_device)
        pipe_3.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split_1,
            split_2,
            scheduler_cls_orig,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
        ):
            inputs = self.get_dummy_inputs(torch_device)
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
            pipe_3.scheduler = scheduler_cls.from_config(pipe_3.scheduler.config)

            # Let's retrieve the number of timesteps we want to use
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()

            split_1_ts = num_train_timesteps - int(round(num_train_timesteps * split_1))
            split_2_ts = num_train_timesteps - int(round(num_train_timesteps * split_2))

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts, expected_steps))
                expected_steps_2 = expected_steps_1[-1:] + list(
                    filter(lambda ts: ts >= split_2_ts and ts < split_1_ts, expected_steps)
                )
                expected_steps_3 = expected_steps_2[-1:] + list(filter(lambda ts: ts < split_2_ts, expected_steps))
                expected_steps = expected_steps_1 + expected_steps_2 + expected_steps_3
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts, expected_steps))
                expected_steps_2 = list(filter(lambda ts: ts >= split_2_ts and ts < split_1_ts, expected_steps))
                expected_steps_3 = list(filter(lambda ts: ts < split_2_ts, expected_steps))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {**inputs, **{"denoising_end": split_1, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]

            assert (
                expected_steps_1 == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

            inputs_2 = {
                **inputs,
                **{"denoising_start": split_1, "denoising_end": split_2, "image": latents, "output_type": "latent"},
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]

            inputs_3 = {**inputs, **{"denoising_start": split_2, "image": latents}}
            pipe_3(**inputs_3).images[0]

            assert expected_steps_3 == done_steps[len(expected_steps_1) + len(expected_steps_2) :]
            assert (
                expected_steps == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

        for steps in [7, 11, 20]:
            for split_1, split_2 in zip([0.19, 0.32], [0.81, 0.68]):
                for scheduler_cls in [
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split_1, split_2, scheduler_cls)

    def test_stable_diffusion_xl_multi_prompts(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(
                0,
                0,
            ),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -3:, -3:, -1]

        assert (
            np.abs(image_slice_with_no_neg_conditions.flatten() - image_slice_with_neg_conditions.flatten()).max()
            > 1e-4
        )

    def test_stable_diffusion_xl_inpaint_mask_latents(self):
        device = "cpu"
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # normal mask + normal image
        ##  `image`: pil, `mask_image``: pil, `masked_image_latents``: None
        inputs = self.get_dummy_inputs(device)
        inputs["strength"] = 0.9
        out_0 = sd_pipe(**inputs).images

        # image latents + mask latents
        inputs = self.get_dummy_inputs(device)
        image = sd_pipe.image_processor.preprocess(inputs["image"]).to(sd_pipe.device)
        mask = sd_pipe.mask_processor.preprocess(inputs["mask_image"]).to(sd_pipe.device)
        masked_image = image * (mask < 0.5)

        generator = torch.Generator(device=device).manual_seed(0)
        image_latents = sd_pipe._encode_vae_image(image, generator=generator)
        torch.randn((1, 4, 32, 32), generator=generator)
        mask_latents = sd_pipe._encode_vae_image(masked_image, generator=generator)
        inputs["image"] = image_latents
        inputs["masked_image_latents"] = mask_latents
        inputs["mask_image"] = mask
        inputs["strength"] = 0.9
        generator = torch.Generator(device=device).manual_seed(0)
        torch.randn((1, 4, 32, 32), generator=generator)
        inputs["generator"] = generator
        out_1 = sd_pipe(**inputs).images
        assert np.abs(out_0 - out_1).max() < 1e-2

    def test_stable_diffusion_xl_inpaint_2_images(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        # test to confirm if we pass two same image, we will get same output
        inputs = self.get_dummy_inputs(device)
        gen1 = torch.Generator(device=device).manual_seed(0)
        gen2 = torch.Generator(device=device).manual_seed(0)
        for name in ["prompt", "image", "mask_image"]:
            inputs[name] = [inputs[name]] * 2
        inputs["generator"] = [gen1, gen2]
        images = sd_pipe(**inputs).images

        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() < 1e-4

        # test to confirm that if we pass two different images, we will get different output
        inputs = self.get_dummy_inputs_2images(device)
        images = sd_pipe(**inputs).images
        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() > 1e-2
