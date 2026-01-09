# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import gc
import tempfile
import unittest

import numpy as np
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    load_image,
    numpy_cosine_similarity_distance,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    SDFunctionTesterMixin,
)


enable_full_determinism()


class StableDiffusionXLPipelineFastTests(
    SDFunctionTesterMixin,
    IPAdapterTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
    unittest.TestCase,
):
    pipeline_class = StableDiffusionXLPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"add_text_embeds", "add_time_ids"})
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(2, 4),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
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
            norm_num_groups=1,
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

        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "image_encoder": None,
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
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_xl_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5388, 0.5452, 0.4694, 0.4583, 0.5253, 0.4832, 0.5288, 0.5035, 0.47])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_euler_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_xl_euler_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_ays(self):
        from diffusers.schedulers import AysSchedules

        timestep_schedule = AysSchedules["StableDiffusionXLTimesteps"]
        sigma_schedule = AysSchedules["StableDiffusionXLSigmas"]

        device = "cpu"  # ensure determinism for the device-dependent torch.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = 10
        output = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = None
        inputs["timesteps"] = timestep_schedule
        output_ts = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs(device)
        inputs["num_inference_steps"] = None
        inputs["sigmas"] = sigma_schedule
        output_sigmas = sd_pipe(**inputs).images

        assert np.abs(output_sigmas.flatten() - output_ts.flatten()).max() < 1e-3, (
            "ays timesteps and ays sigmas should have the same outputs"
        )
        assert np.abs(output.flatten() - output_ts.flatten()).max() > 1e-3, (
            "use ays timesteps should have different outputs"
        )
        assert np.abs(output.flatten() - output_sigmas.flatten()).max() > 1e-3, (
            "use ays sigmas should have different outputs"
        )

    def test_ip_adapter(self):
        expected_pipe_slice = None
        if torch_device == "cpu":
            expected_pipe_slice = np.array([0.5388, 0.5452, 0.4694, 0.4583, 0.5253, 0.4832, 0.5288, 0.5035, 0.4766])

        return super().test_ip_adapter(expected_pipe_slice=expected_pipe_slice)

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    @require_torch_accelerator
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.enable_model_cpu_offload(device=torch_device)
        pipes.append(sd_pipe)

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.enable_sequential_cpu_offload(device=torch_device)
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3
        assert np.abs(image_slices[0] - image_slices[2]).max() < 1e-3

    @unittest.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_two_xl_mixture_of_denoiser_fast(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split,
            scheduler_cls_orig,
            expected_tss,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
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

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split, expected_tss))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split, expected_tss))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split, expected_tss))
                expected_steps_2 = list(filter(lambda ts: ts < split, expected_tss))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {
                **inputs,
                **{
                    "denoising_end": 1.0 - (split / num_train_timesteps),
                    "output_type": "latent",
                },
            }
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {
                **inputs,
                **{
                    "denoising_start": 1.0 - (split / num_train_timesteps),
                    "image": latents,
                },
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        steps = 10
        for split in [300, 700]:
            for scheduler_cls_timesteps in [
                (EulerDiscreteScheduler, [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]),
                (
                    HeunDiscreteScheduler,
                    [
                        901.0,
                        801.0,
                        801.0,
                        701.0,
                        701.0,
                        601.0,
                        601.0,
                        501.0,
                        501.0,
                        401.0,
                        401.0,
                        301.0,
                        301.0,
                        201.0,
                        201.0,
                        101.0,
                        101.0,
                        1.0,
                        1.0,
                    ],
                ),
            ]:
                assert_run_mixture(steps, split, scheduler_cls_timesteps[0], scheduler_cls_timesteps[1])

    @slow
    def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split,
            scheduler_cls_orig,
            expected_tss,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
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

            if pipe_1.scheduler.order == 2:
                expected_steps_1 = list(filter(lambda ts: ts >= split, expected_tss))
                expected_steps_2 = expected_steps_1[-1:] + list(filter(lambda ts: ts < split, expected_tss))
                expected_steps = expected_steps_1 + expected_steps_2
            else:
                expected_steps_1 = list(filter(lambda ts: ts >= split, expected_tss))
                expected_steps_2 = list(filter(lambda ts: ts < split, expected_tss))

            # now we monkey patch step `done_steps`
            # list into the step function for testing
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())  # args[1] is always the passed `t`
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step

            inputs_1 = {
                **inputs,
                **{
                    "denoising_end": 1.0 - (split / num_train_timesteps),
                    "output_type": "latent",
                },
            }
            latents = pipe_1(**inputs_1).images[0]

            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

            inputs_2 = {
                **inputs,
                **{
                    "denoising_start": 1.0 - (split / num_train_timesteps),
                    "image": latents,
                },
            }
            pipe_2(**inputs_2).images[0]

            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        steps = 10
        for split in [300, 500, 700]:
            for scheduler_cls_timesteps in [
                (DDIMScheduler, [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]),
                (EulerDiscreteScheduler, [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]),
                (DPMSolverMultistepScheduler, [901, 811, 721, 631, 541, 451, 361, 271, 181, 91]),
                (UniPCMultistepScheduler, [901, 811, 721, 631, 541, 451, 361, 271, 181, 91]),
                (
                    HeunDiscreteScheduler,
                    [
                        901.0,
                        801.0,
                        801.0,
                        701.0,
                        701.0,
                        601.0,
                        601.0,
                        501.0,
                        501.0,
                        401.0,
                        401.0,
                        301.0,
                        301.0,
                        201.0,
                        201.0,
                        101.0,
                        101.0,
                        1.0,
                        1.0,
                    ],
                ),
            ]:
                assert_run_mixture(steps, split, scheduler_cls_timesteps[0], scheduler_cls_timesteps[1])

        steps = 25
        for split in [300, 500, 700]:
            for scheduler_cls_timesteps in [
                (
                    DDIMScheduler,
                    [
                        961,
                        921,
                        881,
                        841,
                        801,
                        761,
                        721,
                        681,
                        641,
                        601,
                        561,
                        521,
                        481,
                        441,
                        401,
                        361,
                        321,
                        281,
                        241,
                        201,
                        161,
                        121,
                        81,
                        41,
                        1,
                    ],
                ),
                (
                    EulerDiscreteScheduler,
                    [
                        961.0,
                        921.0,
                        881.0,
                        841.0,
                        801.0,
                        761.0,
                        721.0,
                        681.0,
                        641.0,
                        601.0,
                        561.0,
                        521.0,
                        481.0,
                        441.0,
                        401.0,
                        361.0,
                        321.0,
                        281.0,
                        241.0,
                        201.0,
                        161.0,
                        121.0,
                        81.0,
                        41.0,
                        1.0,
                    ],
                ),
                (
                    DPMSolverMultistepScheduler,
                    [
                        951,
                        913,
                        875,
                        837,
                        799,
                        761,
                        723,
                        685,
                        647,
                        609,
                        571,
                        533,
                        495,
                        457,
                        419,
                        381,
                        343,
                        305,
                        267,
                        229,
                        191,
                        153,
                        115,
                        77,
                        39,
                    ],
                ),
                (
                    UniPCMultistepScheduler,
                    [
                        951,
                        913,
                        875,
                        837,
                        799,
                        761,
                        723,
                        685,
                        647,
                        609,
                        571,
                        533,
                        495,
                        457,
                        419,
                        381,
                        343,
                        305,
                        267,
                        229,
                        191,
                        153,
                        115,
                        77,
                        39,
                    ],
                ),
                (
                    HeunDiscreteScheduler,
                    [
                        961.0,
                        921.0,
                        921.0,
                        881.0,
                        881.0,
                        841.0,
                        841.0,
                        801.0,
                        801.0,
                        761.0,
                        761.0,
                        721.0,
                        721.0,
                        681.0,
                        681.0,
                        641.0,
                        641.0,
                        601.0,
                        601.0,
                        561.0,
                        561.0,
                        521.0,
                        521.0,
                        481.0,
                        481.0,
                        441.0,
                        441.0,
                        401.0,
                        401.0,
                        361.0,
                        361.0,
                        321.0,
                        321.0,
                        281.0,
                        281.0,
                        241.0,
                        241.0,
                        201.0,
                        201.0,
                        161.0,
                        161.0,
                        121.0,
                        121.0,
                        81.0,
                        81.0,
                        41.0,
                        41.0,
                        1.0,
                        1.0,
                    ],
                ),
            ]:
                assert_run_mixture(steps, split, scheduler_cls_timesteps[0], scheduler_cls_timesteps[1])

    @slow
    def test_stable_diffusion_three_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
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

            assert expected_steps_1 == done_steps, (
                f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"
            )

            with self.assertRaises(ValueError) as cm:
                inputs_2 = {
                    **inputs,
                    **{
                        "denoising_start": split_2,
                        "denoising_end": split_1,
                        "image": latents,
                        "output_type": "latent",
                    },
                }
                pipe_2(**inputs_2).images[0]
            assert "cannot be larger than or equal to `denoising_end`" in str(cm.exception)

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
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components).to(torch_device)

        # forward with single prompt
        inputs = self.get_dummy_inputs(torch_device)
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        # ensure the results are equal
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs(torch_device)
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -3:, -3:, -1]

        # ensure the results are not equal
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 1e-4

    def test_stable_diffusion_xl_negative_conditions(self):
        device = "cpu"  # ensure determinism for the device-dependent torch.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe = sd_pipe.to(device)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_cond = image[0, -3:, -3:, -1]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_cond = image[0, -3:, -3:, -1]

        self.assertTrue(np.abs(image_slice_with_no_neg_cond - image_slice_with_neg_cond).max() > 1e-2)

    def test_stable_diffusion_xl_save_from_pretrained(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components).to(torch_device)
        pipes.append(sd_pipe)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd_pipe.save_pretrained(tmpdirname)
            sd_pipe = StableDiffusionXLPipeline.from_pretrained(tmpdirname).to(torch_device)
        pipes.append(sd_pipe)

        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()

            inputs = self.get_dummy_inputs(torch_device)
            image = pipe(**inputs).images

            image_slices.append(image[0, -3:, -3:, -1].flatten())

        assert np.abs(image_slices[0] - image_slices[1]).max() < 1e-3

    def test_pipeline_interrupt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe = sd_pipe.to(torch_device)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"
        num_inference_steps = 3

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
            num_inference_steps=num_inference_steps,
            output_type="np",
            generator=torch.Generator("cpu").manual_seed(0),
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
            num_inference_steps=num_inference_steps,
            output_type="latent",
            generator=torch.Generator("cpu").manual_seed(0),
            callback_on_step_end=callback_on_step_end,
        ).images

        # fetch intermediate latents at the interrupted step
        # from the completed generation process
        intermediate_latent = pipe_state.state[interrupt_step_idx]

        # compare the intermediate latent to the output of the interrupted process
        # they should be the same
        assert torch.allclose(intermediate_latent, output_interrupted, atol=1e-4)


@slow
class StableDiffusionXLPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        super().setUp()
        gc.collect()
        backend_empty_cache(torch_device)

    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_stable_diffusion_lcm(self):
        torch.manual_seed(0)
        unet = UNet2DConditionModel.from_pretrained(
            "latent-consistency/lcm-ssd-1b", torch_dtype=torch.float16, variant="fp16"
        )
        sd_pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B", unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to(torch_device)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "a red car standing on the side of the street"

        image = sd_pipe(
            prompt, num_inference_steps=4, guidance_scale=8.0, generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_ssd_1b_lcm.png"
        )

        image = sd_pipe.image_processor.pil_to_numpy(image)
        expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())

        assert max_diff < 1e-2
