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

import copy
import random

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

from ...testing_utils import (
    assert_tensors_close,
    floats_tensor,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ..testing_utils import (
    BasePipelineTesterConfig,
    IPAdapterTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
)


class StableDiffusionXLInpaintPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLInpaintPipeline
    required_input_params_in_call_signature = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_input_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    output_shape = (3, 64, 64)

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

    def get_dummy_inputs(self, seed=0):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed)).to(torch_device)
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        # create mask
        image[8:, 8:, :] = 255
        mask_image = Image.fromarray(np.uint8(image)).convert("L").resize((64, 64))

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": self.get_generator(seed),
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "strength": 1.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs

    def get_dummy_inputs_2images(self, seed=0, img_res=64):
        # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
        image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed)).to(torch_device)
        image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed + 22)).to(torch_device)
        # Convert images to [-1, 1]
        init_image1 = 2.0 * image1 - 1.0
        init_image2 = 2.0 * image2 - 1.0

        # empty mask
        mask_image = torch.zeros((1, 1, img_res, img_res), device=torch_device)

        inputs = {
            "prompt": ["A painting of a squirrel eating a burger"] * 2,
            "image": [init_image1, init_image2],
            "mask_image": [mask_image] * 2,
            "generator": [self.get_generator(seed), self.get_generator(seed)],
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionXLInpaintPipeline(StableDiffusionXLInpaintPipelineTesterConfig, PipelineTesterMixin):
    def test_components_function(self):
        # `requires_aesthetics_score` is a config value, not a component, so it is not part of `pipe.components`.
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.get_pipeline(**init_components)

        assert hasattr(pipe, "components")
        assert set(pipe.components.keys()) == set(init_components.keys())

    def test_stable_diffusion_xl_inpaint_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.7819, 0.5553, 0.5853, 0.6391, 0.6660, 0.7527, 0.6580, 0.5877, 0.5146])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_inpaint_euler_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_inpaint_euler_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 64, 64)

        # Custom timesteps matching the default schedule reproduce `..._euler_lcm`'s output.
        # fmt: off
        expected_slice = torch.tensor([0.6611, 0.5569, 0.5531, 0.5471, 0.5918, 0.6393, 0.5074, 0.5468, 0.5185])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_save_load_optional_components(self, tmp_path, expected_max_difference=1e-4):
        # `_optional_components` also lists the tokenizers and the text encoders, but the standard dummy inputs
        # pass a `prompt`, so those have to stay. Restrict the test to the components that can be dropped.
        droppable_components = ["image_encoder", "feature_extractor"]

        pipe = self.get_pipeline().to(torch_device)
        for optional_component in droppable_components:
            setattr(pipe, optional_component, None)

        torch.manual_seed(0)
        output = pipe(**self.get_dummy_inputs())[0]

        pipe.save_pretrained(tmp_path, safe_serialization=False)
        pipe_loaded = self.pipeline_class.from_pretrained(tmp_path)
        pipe_loaded.to(torch_device)
        pipe_loaded.set_progress_bar_config(disable=None)

        for optional_component in droppable_components:
            assert getattr(pipe_loaded, optional_component) is None, (
                f"`{optional_component}` did not stay set to None after loading."
            )

        torch.manual_seed(0)
        output_loaded = pipe_loaded(**self.get_dummy_inputs())[0]

        assert_tensors_close(
            output_loaded,
            output,
            atol=expected_max_difference,
            msg="Output changed after dropping optional components.",
        )

    def test_stable_diffusion_xl_inpaint_negative_prompt_embeds(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        # forward without prompt embeds
        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with prompt embeds
        inputs = self.get_dummy_inputs()
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
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # make sure that it's equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

    def test_stable_diffusion_xl_refiner(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(skip_first_text_encoder=True))

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.7432, 0.5123, 0.5937, 0.6343, 0.6309, 0.7179, 0.6593, 0.5638, 0.5031])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_two_xl_mixture_of_denoiser_fast(self):
        components = self.get_dummy_components()
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = self.get_pipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs()
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
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = self.get_pipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps, split, scheduler_cls_orig, num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps
        ):
            inputs = self.get_dummy_inputs()
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
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = self.get_pipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = self.get_pipeline(**components).to(torch_device)
        pipe_3.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split_1,
            split_2,
            scheduler_cls_orig,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
        ):
            inputs = self.get_dummy_inputs()
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

            assert expected_steps_1 == done_steps, (
                f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"
            )

            inputs_2 = {
                **inputs,
                **{"denoising_start": split_1, "denoising_end": split_2, "image": latents, "output_type": "latent"},
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]

            inputs_3 = {**inputs, **{"denoising_start": split_2, "image": latents}}
            pipe_3(**inputs_3).images[0]

            assert expected_steps_3 == done_steps[len(expected_steps_1) + len(expected_steps_2) :]
            assert expected_steps == done_steps, (
                f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"
            )

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
        sd_pipe = self.get_pipeline().to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

    def test_stable_diffusion_xl_img2img_negative_conditions(self):
        # Run on CPU: the two runs below are compared against each other.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_conditions = image[0, -1, -3:, -3:]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_conditions = image[0, -1, -3:, -3:]

        assert (image_slice_with_no_neg_conditions - image_slice_with_neg_conditions).abs().max() > 1e-4

    def test_stable_diffusion_xl_inpaint_mask_latents(self):
        # Run on CPU: the VAE-encoded run below is compared against the plain one.
        sd_pipe = self.get_pipeline()

        # normal mask + normal image
        ##  `image`: pil, `mask_image``: pil, `masked_image_latents``: None
        inputs = self.get_dummy_inputs()
        inputs["strength"] = 0.9
        out_0 = sd_pipe(**inputs).images

        # image latents + mask latents
        inputs = self.get_dummy_inputs()
        image = sd_pipe.image_processor.preprocess(inputs["image"]).to(sd_pipe.device)
        mask = sd_pipe.mask_processor.preprocess(inputs["mask_image"]).to(sd_pipe.device)
        masked_image = image * (mask < 0.5)

        generator = self.get_generator(0)
        image_latents = sd_pipe._encode_vae_image(image, generator=generator)
        torch.randn((1, 4, 32, 32), generator=generator)
        mask_latents = sd_pipe._encode_vae_image(masked_image, generator=generator)
        inputs["image"] = image_latents
        inputs["masked_image_latents"] = mask_latents
        inputs["mask_image"] = mask
        inputs["strength"] = 0.9
        generator = self.get_generator(0)
        torch.randn((1, 4, 32, 32), generator=generator)
        inputs["generator"] = generator
        out_1 = sd_pipe(**inputs).images

        assert (out_0 - out_1).abs().max() < 1e-2

    def test_stable_diffusion_xl_inpaint_2_images(self):
        # Run on CPU: the two batch entries below are compared against each other.
        sd_pipe = self.get_pipeline()

        # test to confirm if we pass two same image, we will get same output
        inputs = self.get_dummy_inputs()
        for name in ["prompt", "image", "mask_image"]:
            inputs[name] = [inputs[name]] * 2
        inputs["generator"] = [self.get_generator(0), self.get_generator(0)]
        images = sd_pipe(**inputs).images

        assert images.shape == (2, 3, 64, 64)

        image_slice1 = images[0, -1, -3:, -3:]
        image_slice2 = images[1, -1, -3:, -3:]
        assert (image_slice1 - image_slice2).abs().max() < 1e-4

        # test to confirm that if we pass two different images, we will get different output
        images = sd_pipe(**self.get_dummy_inputs_2images()).images
        assert images.shape == (2, 3, 64, 64)

        image_slice1 = images[0, -1, -3:, -3:]
        image_slice2 = images[1, -1, -3:, -3:]
        assert (image_slice1 - image_slice2).abs().max() > 1e-2

    def test_pipeline_interrupt(self):
        sd_pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()

        prompt = "hey"
        num_inference_steps = 5

        # store intermediate latents from the generation process
        class PipelineState:
            def __init__(self):
                self.state = []

            def apply(self, pipe, i, t, callback_kwargs):
                self.state.append(callback_kwargs["latents"])
                return callback_kwargs

        pipe_state = PipelineState()
        sd_pipe(
            prompt,
            image=inputs["image"],
            mask_image=inputs["mask_image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            generator=self.get_generator(0),
            callback_on_step_end=pipe_state.apply,
        ).images

        # interrupt generation at step index
        interrupt_step_idx = 1

        def callback_on_step_end(pipe, i, t, callback_kwargs):
            if i == interrupt_step_idx:
                pipe._interrupt = True

            return callback_kwargs

        output_interrupted = sd_pipe(
            prompt,
            image=inputs["image"],
            mask_image=inputs["mask_image"],
            strength=0.8,
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=self.get_generator(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert_tensors_close(intermediate_latent, output_interrupted, atol=1e-4)


class TestStableDiffusionXLInpaintPipelineMemory(StableDiffusionXLInpaintPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL inpaint pipeline."""


class TestStableDiffusionXLInpaintPipelineIPAdapter(
    StableDiffusionXLInpaintPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL inpaint pipeline."""
