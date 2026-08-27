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

import numpy as np
import pytest
import torch
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D
from diffusers.pipelines.controlnet.pipeline_controlnet import MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import (
    assert_tensors_close,
    backend_empty_cache,
    enable_full_determinism,
    load_image,
    require_torch_accelerator,
    slow,
    torch_device,
)
from ..pipeline_params import TEXT_TO_IMAGE_BATCH_PARAMS, TEXT_TO_IMAGE_PARAMS
from ..stable_diffusion.ip_adapter_tester import IPAdapterTesterMixin
from ..testing_utils import BasePipelineTesterConfig, MemoryTesterMixin, PipelineTesterMixin


enable_full_determinism()


class StableDiffusionXLControlNetPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 64, 64)

    def get_dummy_components(self, time_cond_proj_dim=None):
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
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)
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

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        # The conditioning image is drawn from the same generator that is handed to the pipeline, so the pipeline
        # sees an already-advanced generator state — keep the order to stay comparable with the expected slices.
        generator = self.get_generator(0)
        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=torch.device(torch_device),
        )

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": image,
        }


class StableDiffusionXLControlNetPipelineTests:
    """Tests shared by the SDXL ControlNet config and the SSD-1B one, which only differ in their components.

    The two expected slices below are the only per-config values, so subclasses set them instead of overriding the
    tests.
    """

    expected_guess_slice: torch.Tensor
    expected_lcm_slice: torch.Tensor

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_multi_prompts(self):
        pipe = self.get_pipeline().to(torch_device)

        # forward with single prompt
        image_slice_1 = pipe(**self.get_dummy_inputs()).images[0, -1, -3:, -3:]

        # forward with same prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = inputs["prompt"]
        image_slice_2 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different prompt
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "different prompt"
        image_slice_3 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

        # manually set a negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        image_slice_1 = pipe(**inputs).images[0, -1, -3:, -3:]

        # forward with same negative_prompt duplicated
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        image_slice_2 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are equal
        assert (image_slice_1 - image_slice_2).abs().max() < 1e-4

        # forward with different negative_prompt
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        image_slice_3 = pipe(**inputs).images[0, -1, -3:, -3:]

        # ensure the results are not equal
        assert (image_slice_1 - image_slice_3).abs().max() > 1e-4

    def test_controlnet_sdxl_guess(self):
        # Run on CPU: the expected slice is CPU-specific.
        pipe = self.get_pipeline()

        inputs = self.get_dummy_inputs()
        inputs["guess_mode"] = True

        image_slice = pipe(**inputs).images[0, -1, -3:, -3:]

        # make sure that it's equal
        assert_tensors_close(image_slice.flatten().cpu(), self.expected_guess_slice, atol=1e-4)

    def test_controlnet_sdxl_lcm(self):
        pipe = self.get_pipeline(**self.get_dummy_components(time_cond_proj_dim=256))
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(torch_device)

        image = pipe(**self.get_dummy_inputs()).images
        assert image.shape == (1, *self.output_shape)

        image_slice = image[0, -1, -3:, -3:]
        assert_tensors_close(image_slice.flatten().cpu(), self.expected_lcm_slice, atol=1e-2)


class TestStableDiffusionXLControlNetPipeline(
    StableDiffusionXLControlNetPipelineTesterConfig, StableDiffusionXLControlNetPipelineTests, PipelineTesterMixin
):
    # fmt: off
    expected_guess_slice = torch.tensor([0.715316, 0.563373, 0.569716, 0.620860, 0.569999, 0.604369, 0.428272, 0.455195, 0.527119])
    expected_lcm_slice = torch.tensor([0.7820, 0.6195, 0.6193, 0.7045, 0.6706, 0.5837, 0.4147, 0.5232, 0.4868])
    # fmt: on

    # Copied from test_stable_diffusion_xl.py:test_stable_diffusion_two_xl_mixture_of_denoiser_fast
    # with `StableDiffusionXLControlNetPipeline` instead of `StableDiffusionXLPipeline`
    def test_controlnet_sdxl_two_mixture_of_denoiser_fast(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLControlNetPipeline(**components).to(torch_device)
        pipe_1.unet.set_default_attn_processor()

        components_without_controlnet = {k: v for k, v in components.items() if k != "controlnet"}
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components_without_controlnet).to(torch_device)
        pipe_2.unet.set_default_attn_processor()

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


class TestStableDiffusionXLControlNetPipelineIPAdapter(
    StableDiffusionXLControlNetPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SDXL ControlNet pipeline."""


class TestStableDiffusionXLControlNetPipelineMemory(
    StableDiffusionXLControlNetPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SDXL ControlNet pipeline."""


class StableDiffusionXLMultiControlNetPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 64, 64)

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
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(1.0)

        controlnet1 = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        controlnet1.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
        controlnet2 = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        controlnet2.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
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

        controlnet = MultiControlNetModel([controlnet1, controlnet2])

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        # The conditioning images are drawn from the same generator that is handed to the pipeline, so the pipeline
        # sees an already-advanced generator state — keep the order to stay comparable across runs.
        generator = self.get_generator(0)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(torch_device),
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(torch_device),
            ),
        ]

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": images,
        }


class TestStableDiffusionXLMultiControlNetPipeline(
    StableDiffusionXLMultiControlNetPipelineTesterConfig, PipelineTesterMixin
):
    def test_control_guidance_switch(self):
        pipe = self.get_pipeline().to(torch_device)

        scale = 10.0
        steps = 4

        def run(**extra):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = steps
            inputs["controlnet_conditioning_scale"] = scale
            return pipe(**inputs, **extra)[0]

        output_1 = run()
        output_2 = run(control_guidance_start=0.1, control_guidance_end=0.2)
        output_3 = run(control_guidance_start=[0.1, 0.3], control_guidance_end=[0.2, 0.7])
        output_4 = run(control_guidance_start=0.4, control_guidance_end=[0.5, 0.8])

        # make sure that all outputs are different
        assert (output_1 - output_2).abs().sum() > 1e-3
        assert (output_1 - output_3).abs().sum() > 1e-3
        assert (output_1 - output_4).abs().sum() > 1e-3

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass


class StableDiffusionXLMultiControlNetOneModelPipelineTesterConfig(BasePipelineTesterConfig):
    pipeline_class = StableDiffusionXLControlNetPipeline
    required_input_params_in_call_signature = TEXT_TO_IMAGE_PARAMS
    batch_input_params = TEXT_TO_IMAGE_BATCH_PARAMS
    output_shape = (3, 64, 64)

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
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight)
                m.bias.data.fill_(1.0)

        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        controlnet.controlnet_down_blocks.apply(init_weights)

        torch.manual_seed(0)
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

        controlnet = MultiControlNetModel([controlnet])

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }

    def get_dummy_inputs(self):
        # The conditioning image is drawn from the same generator that is handed to the pipeline, so the pipeline
        # sees an already-advanced generator state — keep the order to stay comparable across runs.
        generator = self.get_generator(0)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=torch.device(torch_device),
            ),
        ]

        return {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Request torch outputs so tests compare torch tensors directly (see `BasePipelineTesterConfig`).
            # Note `"pt"` images are `(batch, channels, height, width)`, unlike `"np"` (`(batch, h, w, c)`).
            "output_type": "pt",
            "image": images,
        }


class TestStableDiffusionXLMultiControlNetOneModelPipeline(
    StableDiffusionXLMultiControlNetOneModelPipelineTesterConfig, PipelineTesterMixin
):
    def test_control_guidance_switch(self):
        pipe = self.get_pipeline().to(torch_device)

        scale = 10.0
        steps = 4

        def run(**extra):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = steps
            inputs["controlnet_conditioning_scale"] = scale
            return pipe(**inputs, **extra)[0]

        output_1 = run()
        output_2 = run(control_guidance_start=0.1, control_guidance_end=0.2)
        output_3 = run(control_guidance_start=[0.1], control_guidance_end=[0.2])
        output_4 = run(control_guidance_start=0.4, control_guidance_end=[0.5])

        # make sure that all outputs are different
        assert (output_1 - output_2).abs().sum() > 1e-3
        assert (output_1 - output_3).abs().sum() > 1e-3
        assert (output_1 - output_4).abs().sum() > 1e-3

    def test_inference_batch_single_identical(self, batch_size=3, expected_max_diff=2e-3):
        super().test_inference_batch_single_identical(batch_size=batch_size, expected_max_diff=expected_max_diff)

    @pytest.mark.skip("We test this functionality elsewhere already.")
    def test_save_load_optional_components(self):
        pass

    def test_negative_conditions(self):
        pipe = self.get_pipeline().to(torch_device)

        inputs = self.get_dummy_inputs()
        image_slice_without_neg_cond = pipe(**inputs).images[0, -1, -3:, -3:]

        image_slice_with_neg_cond = pipe(
            **inputs,
            negative_original_size=(512, 512),
            negative_crops_coords_top_left=(0, 0),
            negative_target_size=(1024, 1024),
        ).images[0, -1, -3:, -3:]

        assert (image_slice_without_neg_cond - image_slice_with_neg_cond).abs().max() > 1e-2


@slow
@require_torch_accelerator
class TestControlNetSDXLPipelineSlow:
    @pytest.fixture(autouse=True)
    def cleanup(self):
        gc.collect()
        backend_empty_cache(torch_device)
        yield
        gc.collect()
        backend_empty_cache(torch_device)

    def test_canny(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet
        )
        pipe.enable_sequential_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4185, 0.4127, 0.4089, 0.4046, 0.4115, 0.4096, 0.4081, 0.4112, 0.3913])
        assert np.allclose(original_image, expected_image, atol=1e-04)

    def test_depth(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0")

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet
        )
        pipe.enable_sequential_cpu_offload(device=torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3).images

        assert images[0].shape == (512, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4399, 0.5112, 0.5478, 0.4314, 0.472, 0.4823, 0.4647, 0.4957, 0.4853])
        assert np.allclose(original_image, expected_image, atol=1e-04)


class StableDiffusionSSD1BControlNetPipelineTesterConfig(StableDiffusionXLControlNetPipelineTesterConfig):
    """Same contract as the SDXL ControlNet config, but built on an SSD-1B-shaped UNet / ControlNet."""

    def get_dummy_components(self, time_cond_proj_dim=None):
        torch.manual_seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
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
        torch.manual_seed(0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            mid_block_type="UNetMidBlock2D",
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
        )
        torch.manual_seed(0)
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

        return {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "feature_extractor": None,
            "image_encoder": None,
        }


class TestStableDiffusionSSD1BControlNetPipeline(
    StableDiffusionSSD1BControlNetPipelineTesterConfig, StableDiffusionXLControlNetPipelineTests, PipelineTesterMixin
):
    # fmt: off
    expected_guess_slice = torch.tensor([0.669912, 0.557802, 0.523260, 0.596366, 0.552897, 0.576922, 0.433411, 0.450273, 0.491615])
    expected_lcm_slice = torch.tensor([0.6787, 0.5117, 0.5558, 0.6963, 0.6571, 0.5928, 0.4121, 0.5468, 0.5057])
    # fmt: on

    def test_conditioning_channels(self):
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            # SD2-specific config below
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
            cross_attention_dim=64,
            time_cond_proj_dim=None,
        )

        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=4)
        assert type(controlnet.mid_block) is UNetMidBlock2D
        assert controlnet.conditioning_channels == 4


class TestStableDiffusionSSD1BControlNetPipelineIPAdapter(
    StableDiffusionSSD1BControlNetPipelineTesterConfig, IPAdapterTesterMixin
):
    """IP-Adapter tests for the SSD-1B ControlNet pipeline."""


class TestStableDiffusionSSD1BControlNetPipelineMemory(
    StableDiffusionSSD1BControlNetPipelineTesterConfig, MemoryTesterMixin
):
    """Memory optimization tests (CPU offload, group offload, layerwise casting) for the SSD-1B ControlNet pipeline."""
