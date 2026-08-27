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
import gc
import importlib
import os
import time

import numpy as np
import pytest
import torch
from packaging import version
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    AutoPipelineForText2Image,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import logging

from ...models.testing_utils.lora import check_if_lora_correctly_set
from ...testing_utils import (
    CaptureLogger,
    assert_tensors_close,
    backend_empty_cache,
    backend_synchronize,
    load_image,
    nightly,
    numpy_cosine_similarity_distance,
    require_peft_backend,
    require_torch_accelerator,
    require_torch_neuron,
    skip_mps,
    slow,
    torch_device,
)
from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..stable_diffusion.ip_adapter_tester import IPAdapterTesterMixin
from ..testing_utils import (
    BasePipelineTesterConfig,
    LoraMemoryTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    PipelineTesterMixin,
    UNetLoraTesterMixin,
    check_qkv_fusion_matches_attn_procs_length,
    check_qkv_fusion_processors_exist,
)


def state_dicts_almost_equal(sd1, sd2):
    sd1 = dict(sorted(sd1.items()))
    sd2 = dict(sorted(sd2.items()))

    for ten1, ten2 in zip(sd1.values(), sd2.values()):
        if (ten1 - ten2).abs().max() > 1e-3:
            return False

    return True


class StableDiffusionXLPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 64, 64)

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

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": self.get_generator(0),
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
        }
        return inputs


class TestStableDiffusionXLPipeline(StableDiffusionXLPipelineTesterConfig, PipelineTesterMixin):
    def test_stable_diffusion_xl_euler(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline()

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.5556, 0.5318, 0.4649, 0.4321, 0.4824, 0.4624, 0.5183, 0.4981, 0.4692])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_euler_lcm(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        image = sd_pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, 3, 64, 64)

        # fmt: off
        expected_slice = torch.tensor([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_xl_euler_lcm_custom_timesteps(self):
        # Run on CPU: the expected slice below is CPU-specific.
        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        assert image.shape == (1, 3, 64, 64)

        # Custom timesteps matching the default 2-step schedule reproduce `test_stable_diffusion_xl_euler_lcm`.
        # fmt: off
        expected_slice = torch.tensor([0.4917, 0.6555, 0.4348, 0.5219, 0.7324, 0.4855, 0.5168, 0.5447, 0.5156])
        # fmt: on
        assert_tensors_close(image[0, -1, -3:, -3:].flatten(), expected_slice, atol=1e-2)

    def test_stable_diffusion_ays(self):
        from diffusers.schedulers import AysSchedules

        timestep_schedule = AysSchedules["StableDiffusionXLTimesteps"]
        sigma_schedule = AysSchedules["StableDiffusionXLSigmas"]

        sd_pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256)).to(torch_device)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 10
        output = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = None
        inputs["timesteps"] = timestep_schedule
        output_ts = sd_pipe(**inputs).images

        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = None
        inputs["sigmas"] = sigma_schedule
        output_sigmas = sd_pipe(**inputs).images

        assert (output_sigmas - output_ts).abs().max() < 1e-3, (
            "ays timesteps and ays sigmas should have the same outputs"
        )
        assert (output - output_ts).abs().max() > 1e-3, "use ays timesteps should have different outputs"
        assert (output - output_sigmas).abs().max() > 1e-3, "use ays sigmas should have different outputs"

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

    def test_stable_diffusion_xl_vae_slicing(self, image_count=4):
        # Run on CPU: sliced VAE decoding is compared against a full-batch decode of the same run.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = sd_pipe(**inputs)

        # make sure sliced vae decode yields the same result
        sd_pipe.vae.enable_slicing()
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = sd_pipe(**inputs)

        # there is a small discrepancy at image borders vs. full batch decode
        assert (output_2.images - output_1.images).abs().max() < 1e-2

    def test_stable_diffusion_xl_vae_tiling(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        # Tiled decode should yield the same result as the non-tiled decode cached by `base_pipe_output`.
        sd_pipe.vae.enable_tiling()
        torch.manual_seed(0)
        output_tiled = sd_pipe(**self.get_dummy_inputs()).images

        assert (output_tiled - base_pipe_output).abs().max() < 5e-1

        # test that tiled decode works with various shapes
        with torch.no_grad():
            for shape in [(1, 4, 73, 97), (1, 4, 65, 49)]:
                sd_pipe.vae.decode(torch.zeros(shape, device=torch_device))

    # MPS currently doesn't support ComplexFloats, which are required for freeU - see https://github.com/huggingface/diffusers/issues/7569.
    @skip_mps
    def test_freeu_enabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        torch.manual_seed(0)
        output_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert not torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_freeu[0, -1, -3:, -3:]), (
            "Enabling of FreeU should lead to different results."
        )

    def test_freeu_disabled(self, base_pipe_output):
        sd_pipe = self.get_pipeline().to(torch_device)

        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        sd_pipe.disable_freeu()

        for upsample_block in sd_pipe.unet.up_blocks:
            for key in {"s1", "s2", "b1", "b2"}:
                assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

        torch.manual_seed(0)
        output_no_freeu = sd_pipe(**self.get_dummy_inputs()).images

        assert torch.allclose(base_pipe_output[0, -1, -3:, -3:], output_no_freeu[0, -1, -3:, -3:], atol=1e-2), (
            "Disabling of FreeU should lead to results similar to the default pipeline results."
        )

    def test_fused_qkv_projections(self, base_pipe_output):
        # The unfused reference is the class-cached `base_pipe_output`, so the runs below reseed the global RNG to
        # reproduce it and the only remaining difference comes from the projection fusion itself.
        sd_pipe = self.get_pipeline().to(torch_device)

        original_image_slice = base_pipe_output[0, -1, -3:, -3:]

        sd_pipe.fuse_qkv_projections()
        for component in sd_pipe.components.values():
            if (
                isinstance(component, torch.nn.Module)
                and getattr(component, "original_attn_processors", None) is not None
            ):
                assert check_qkv_fusion_processors_exist(component), (
                    "Something wrong with the fused attention processors. Expected all the attention processors to be fused."
                )
                assert check_qkv_fusion_matches_attn_procs_length(component, component.original_attn_processors), (
                    "Something wrong with the attention processors concerning the fused QKV projections."
                )

        torch.manual_seed(0)
        image_slice_fused = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        sd_pipe.unfuse_qkv_projections()
        torch.manual_seed(0)
        image_slice_disabled = sd_pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        assert torch.allclose(original_image_slice, image_slice_fused, atol=1e-2, rtol=1e-2), (
            "Fusion of QKV projections shouldn't affect the outputs."
        )
        assert torch.allclose(image_slice_fused, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Outputs, with QKV projection fusion enabled, shouldn't change when fused QKV projections are disabled."
        )
        assert torch.allclose(original_image_slice, image_slice_disabled, atol=1e-2, rtol=1e-2), (
            "Original outputs should match when fused QKV projections are disabled."
        )

    def test_stable_diffusion_two_xl_mixture_of_denoiser_fast(self):
        components = self.get_dummy_components()
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_2.set_progress_bar_config(disable=None)

        def assert_run_mixture(
            num_steps,
            split,
            scheduler_cls_orig,
            expected_tss,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
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
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_2.set_progress_bar_config(disable=None)

        def assert_run_mixture(
            num_steps,
            split,
            scheduler_cls_orig,
            expected_tss,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
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
        pipe_1 = self.get_pipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_2.unet.set_default_attn_processor()
        pipe_2.set_progress_bar_config(disable=None)
        pipe_3 = StableDiffusionXLImg2ImgPipeline(**components).to(torch_device)
        pipe_3.unet.set_default_attn_processor()
        pipe_3.set_progress_bar_config(disable=None)

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

            with pytest.raises(ValueError) as error:
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
            assert "cannot be larger than or equal to `denoising_end`" in str(error.value)

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
        output = sd_pipe(**self.get_dummy_inputs())
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -1, -3:, -3:]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

    def test_stable_diffusion_xl_negative_conditions(self):
        # Run on CPU: the two runs below are compared against each other.
        sd_pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice_with_no_neg_cond = image[0, -1, -3:, -3:]

        image = sd_pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images
        image_slice_with_neg_cond = image[0, -1, -3:, -3:]

        assert (image_slice_with_no_neg_cond - image_slice_with_neg_cond).abs().max() > 1e-2

    def test_pipeline_interrupt(self):
        sd_pipe = self.get_pipeline().to(torch_device)

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


class TestStableDiffusionXLPipelineMemory(StableDiffusionXLPipelineTesterConfig, MemoryTesterMixin):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL pipeline."""


class TestStableDiffusionXLPipelineIPAdapter(StableDiffusionXLPipelineTesterConfig, IPAdapterTesterMixin):
    """IP-Adapter tests for the SDXL pipeline."""


class TestStableDiffusionXLPipelineLoRA(StableDiffusionXLPipelineTesterConfig, LoraTesterMixin, UNetLoraTesterMixin):
    """LoRA tests for the SDXL pipeline."""


class TestStableDiffusionXLPipelineLoRAMemory(StableDiffusionXLPipelineTesterConfig, LoraMemoryTesterMixin):
    """LoRA x offloading tests for the SDXL pipeline."""


@slow
class TestStableDiffusionXLPipelineIntegration:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
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


@slow
@nightly
@require_torch_accelerator
@require_peft_backend
class TestStableDiffusionXLLoRAIntegration:
    """LoRA integration tests against the released SDXL checkpoints.

    The ControlNet and T2I-Adapter cases live here too: they exercise SDXL LoRA checkpoints, only through a
    pipeline that wraps the same SDXL UNet.
    """

    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_sdxl_1_0_lora(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_sdxl_1_0_blockwise_lora(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, adapter_name="offset")
        scales = {
            "unet": {
                "down": {"block_1": [1.0, 1.0], "block_2": [1.0, 1.0]},
                "mid": 1.0,
                "up": {"block_0": [1.0, 1.0, 1.0], "block_1": [1.0, 1.0, 1.0]},
            },
        }
        pipe.set_adapters(["offset"], [scales])

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([00.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_sdxl_lcm_lora(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdxl"

        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "masterpiece, best quality, mountain", generator=generator, num_inference_steps=4, guidance_scale=0.5
        ).images[0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdxl_lcm_lora.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_sdxl_1_0_lora_fusion(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        pipe.fuse_lora()
        # We need to unload the lora weights since in the previous API `fuse_lora` led to lora weights being
        # silently deleted - otherwise this will CPU OOM
        pipe.unload_lora_weights()

        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        # This way we also test equivalence between LoRA fusion and the non-fusion behaviour.
        expected = np.array([0.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

    def test_sdxl_1_0_lora_unfusion(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        pipe.enable_model_cpu_offload()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=3
        ).images
        images_with_fusion = images.flatten()

        pipe.unfuse_lora()
        generator = torch.Generator("cpu").manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=3
        ).images
        images_without_fusion = images.flatten()

        max_diff = numpy_cosine_similarity_distance(images_with_fusion, images_without_fusion)
        assert max_diff < 1e-4

    def test_sdxl_1_0_lora_unfusion_effectivity(self):
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        original_image_slice = images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        _ = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        pipe.unfuse_lora()

        # We need to unload the lora weights - in the old API unfuse led to unloading the adapter weights
        pipe.unload_lora_weights()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        images_without_fusion_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(images_without_fusion_slice, original_image_slice)
        assert max_diff < 1e-3

    def test_sdxl_1_0_lora_fusion_efficiency(self):
        generator = torch.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, torch_dtype=torch.float16)
        pipe.enable_model_cpu_offload()

        start_time = time.time()
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_non_fusion = end_time - start_time

        del pipe

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, torch_dtype=torch.float16)
        pipe.fuse_lora()

        # We need to unload the lora weights since in the previous API `fuse_lora` led to lora weights being
        # silently deleted - otherwise this will CPU OOM
        pipe.unload_lora_weights()
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(0)
        start_time = time.time()
        for _ in range(3):
            pipe(
                "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
            ).images
        end_time = time.time()
        elapsed_time_fusion = end_time - start_time

        assert elapsed_time_fusion < elapsed_time_non_fusion

    def test_sdxl_1_0_last_ben(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_model_cpu_offload()
        lora_model_id = "TheLastBen/Papercut_SDXL"
        lora_filename = "papercut.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe("papercut.safetensors", output_type="np", generator=generator, num_inference_steps=2).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.5244, 0.4347, 0.4312, 0.4246, 0.4398, 0.4409, 0.4884, 0.4938, 0.4094])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_sdxl_1_0_fuse_unfuse_all(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        text_encoder_1_sd = copy.deepcopy(pipe.text_encoder.state_dict())
        text_encoder_2_sd = copy.deepcopy(pipe.text_encoder_2.state_dict())
        unet_sd = copy.deepcopy(pipe.unet.state_dict())

        pipe.load_lora_weights(
            "davizca87/sun-flower", weight_name="snfw3rXL-000004.safetensors", torch_dtype=torch.float16
        )

        fused_te_state_dict = pipe.text_encoder.state_dict()
        fused_te_2_state_dict = pipe.text_encoder_2.state_dict()
        unet_state_dict = pipe.unet.state_dict()

        peft_ge_070 = version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0")

        def remap_key(key, sd):
            # some keys have moved around for PEFT >= 0.7.0, but they should still be loaded correctly
            if (key in sd) or (not peft_ge_070):
                return key

            # instead of linear.weight, we now have linear.base_layer.weight, etc.
            if key.endswith(".weight"):
                key = key[:-7] + ".base_layer.weight"
            elif key.endswith(".bias"):
                key = key[:-5] + ".base_layer.bias"
            return key

        for key, value in text_encoder_1_sd.items():
            key = remap_key(key, fused_te_state_dict)
            assert torch.allclose(fused_te_state_dict[key], value)

        for key, value in text_encoder_2_sd.items():
            key = remap_key(key, fused_te_2_state_dict)
            assert torch.allclose(fused_te_2_state_dict[key], value)

        for key, value in unet_state_dict.items():
            assert torch.allclose(unet_state_dict[key], value)

        pipe.fuse_lora()
        pipe.unload_lora_weights()

        assert not state_dicts_almost_equal(text_encoder_1_sd, pipe.text_encoder.state_dict())
        assert not state_dicts_almost_equal(text_encoder_2_sd, pipe.text_encoder_2.state_dict())
        assert not state_dicts_almost_equal(unet_sd, pipe.unet.state_dict())

        del unet_sd, text_encoder_1_sd, text_encoder_2_sd

    def test_sdxl_1_0_lora_with_sequential_cpu_offloading(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        pipe.enable_sequential_cpu_offload()
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_controlnet_canny_lora(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet
        )
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")
        pipe.enable_sequential_cpu_offload()

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "corgi"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4574, 0.4487, 0.4435, 0.5163, 0.4396, 0.4411, 0.518, 0.4465, 0.4333])

        max_diff = numpy_cosine_similarity_distance(expected_image, original_image)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    @nightly
    def test_sequential_fuse_unfuse(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )

        # 1. round
        pipe.load_lora_weights("Pclanglais/TintinIA", torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        image_slice = images[0, -3:, -3:, -1].flatten()

        pipe.unfuse_lora()

        # 2. round
        pipe.load_lora_weights("ProomptEngineer/pe-balloon-diffusion-style", torch_dtype=torch.float16)
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 3. round
        pipe.load_lora_weights("ostris/crayon_style_lora_sdxl", torch_dtype=torch.float16)
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 4. back to 1st round
        pipe.load_lora_weights("Pclanglais/TintinIA", torch_dtype=torch.float16)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images_2 = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        ).images
        image_slice_2 = images_2[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(image_slice, image_slice_2)
        assert max_diff < 1e-3
        pipe.unload_lora_weights()

    @nightly
    def test_integration_logits_multi_adapter(self):
        path = "stabilityai/stable-diffusion-xl-base-1.0"
        lora_id = "CiroN2022/toy-face"

        pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=torch.float16)
        pipe.load_lora_weights(lora_id, weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipe = pipe.to(torch_device)

        assert check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet"

        prompt = "toy_face of a hacker with a hoodie"

        lora_scale = 0.9

        images = pipe(
            prompt=prompt,
            num_inference_steps=30,
            generator=torch.manual_seed(0),
            cross_attention_kwargs={"scale": lora_scale},
            output_type="np",
        ).images
        expected_slice_scale = np.array([0.538, 0.539, 0.540, 0.540, 0.542, 0.539, 0.538, 0.541, 0.539])

        predicted_slice = images[0, -3:, -3:, -1].flatten()
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipe.set_adapters("pixel")

        prompt = "pixel art, a hacker with a hoodie, simple, flat colors"
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images

        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array(
            [0.61973065, 0.62018543, 0.62181497, 0.61933696, 0.6208608, 0.620576, 0.6200281, 0.62258327, 0.6259889]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        # multi-adapter inference
        pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": 1.0},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images
        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array([0.5888, 0.5897, 0.5946, 0.5888, 0.5935, 0.5946, 0.5857, 0.5891, 0.5909])
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        # Lora disabled
        pipe.disable_lora()
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0),
            output_type="np",
        ).images
        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array([0.5456, 0.5466, 0.5487, 0.5458, 0.5469, 0.5454, 0.5446, 0.5479, 0.5487])
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

    @nightly
    def test_integration_logits_for_dora_lora(self):
        pipeline = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

        logger = logging.get_logger("diffusers.loaders.lora_pipeline")
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            pipeline.load_lora_weights("hf-internal-testing/dora-trained-on-kohya")
            pipeline.enable_model_cpu_offload()
            images = pipeline(
                "photo of ohwx dog",
                num_inference_steps=10,
                generator=torch.manual_seed(0),
                output_type="np",
            ).images
        assert "It seems like you are using a DoRA checkpoint" in cap_logger.out

        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array([0.1817, 0.0697, 0.2346, 0.0900, 0.1261, 0.2279, 0.1767, 0.1991, 0.2886])
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3


@require_torch_neuron
class TestStableDiffusionXLTurboPipelineIntegration:
    ckpt_id = "stabilityai/sdxl-turbo"
    prompt = "A small cactus with a happy face in the Sahara desert."

    @pytest.fixture(autouse=True)
    def cleanup(self):
        saved_env = {"TORCH_NEURONX_ENABLE_NKI_SDPA": os.environ.get("TORCH_NEURONX_ENABLE_NKI_SDPA")}
        os.environ.setdefault("TORCH_NEURONX_ENABLE_NKI_SDPA", "0")
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
        gc.collect()
        backend_empty_cache(torch_device)

    def test_sdxl_turbo_512(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = AutoPipelineForText2Image.from_pretrained(self.ckpt_id, torch_dtype=torch.float16, variant="fp16")
        pipe.to(torch_device)
        backend_synchronize(torch_device)
        pipe.set_progress_bar_config(disable=None)

        image = pipe(
            self.prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
            output_type="np",
        ).images

        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 512, 512, 3)
        assert np.all((image >= 0.0) & (image <= 1.0)), "Pixel values must be in [0, 1]"
        expected_slice = np.array([0.3524, 0.3160, 0.3652, 0.3316, 0.3376, 0.3315, 0.3042, 0.3102, 0.3449])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 5e-2

    @require_torch_neuron
    def test_sdxl_turbo_neuron_compile_256(self):
        from torch_neuronx.neuron_dynamo_backend import set_model_name
        from transformers.utils.output_capturing import install_all_output_capturing_hooks

        device = torch.neuron.current_device()
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = AutoPipelineForText2Image.from_pretrained(self.ckpt_id, torch_dtype=torch.bfloat16, variant="fp16")
        pipe = pipe.to(device)
        backend_synchronize(torch_device)

        pipe.unet.eval()
        pipe.vae.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()

        install_all_output_capturing_hooks(pipe.text_encoder)
        set_model_name("sdxl_turbo_text_encoder")
        pipe.text_encoder = torch.compile(pipe.text_encoder, backend="neuron", fullgraph=True)

        install_all_output_capturing_hooks(pipe.text_encoder_2)
        set_model_name("sdxl_turbo_text_encoder_2")
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, backend="neuron", fullgraph=True)

        set_model_name("sdxl_turbo_unet")
        pipe.unet = torch.compile(pipe.unet, backend="neuron", fullgraph=True)

        # Pre-warm text encoders and copy ops for 256×256 (latent: 32×32).
        tok_kwargs = {"padding": "max_length", "max_length": 77, "truncation": True, "return_tensors": "pt"}
        with torch.no_grad():
            _ids = pipe.tokenizer("warmup", **tok_kwargs).input_ids.to(device)
            _ = pipe.text_encoder(_ids, output_hidden_states=True)
            _ids2 = pipe.tokenizer_2("warmup", **tok_kwargs).input_ids.to(device)
            _ = pipe.text_encoder_2(_ids2, output_hidden_states=True)
            for _shape, _dtype in [((1, 4, 32, 32), torch.bfloat16), ((1, 6), torch.bfloat16)]:
                _ = torch.zeros(_shape, dtype=_dtype).to(device)
        backend_synchronize(torch_device)

        image = pipe(
            self.prompt,
            height=256,
            width=256,
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=generator,
            output_type="np",
        ).images

        assert image.shape == (1, 256, 256, 3)
        assert not np.isnan(image).any(), "Output contains NaN values"
        assert (image >= 0.0).all() and (image <= 1.0).all(), "Output pixel values outside [0, 1]"
